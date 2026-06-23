#!/usr/bin/env python
"""Generate the GROUND-TRUTH molecular continuous opacity reference (READ-ONLY).

Runs pykurucz's own CHOP / OHOP / H2-CIA (Borysow) routines on the shipped m3500
cool-dwarf atmosphere and writes:

  reference/mol_continuum_inputs.npz   -- SHIPPED INPUTS for the from-scratch code:
        T per depth, mass density, H ground-state pop + bhyd, He I pop,
        the CH and OH molecular number-density mode-1 populations (from the
        molecular equilibrium, taken here as INPUT), the wavelength grid, and
        the CH/OH/Borysow-H2 coefficient TABLES.

  reference/mol_continuum_truth.npz    -- GROUND TRUTH (pykurucz output, for
        comparison only): per-species CHOP / OHOP / H2-CIA opacity (cm^2/g),
        each shape (n_layers, nfreq).

This script imports pykurucz (READ-ONLY); the from-scratch verifier
(verify_mol_continuum.py) imports NOTHING from pykurucz.
"""
import sys
import numpy as np
from pathlib import Path

PYK = "/Users/ysting/pykurucz"
sys.path.insert(0, PYK)

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"
ATM_NPZ = REF / "m3500g50.npz"
MOL_DAT = REF / "molecules.dat"

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics.mol_populations import compute_mol_xnfpmol_dopple
from synthe_py.physics import kapp as K

# ── 1. Load the cool-dwarf atmosphere ────────────────────────────────────────
atm = load_cached(ATM_NPZ)
temp = np.asarray(atm.temperature, dtype=np.float64)
rho = np.maximum(np.asarray(atm.mass_density, dtype=np.float64), 1e-30)
n_layers = atm.layers

# Ground-state H population and H partition function ground term (bhyd[:,0]),
# He I population.  These are EOS inputs already stored in the atmosphere npz.
xnfph1 = np.asarray(atm.xnfph, dtype=np.float64)[:, 0]   # XNFPH(:,1): H ground-state mode-11
bhyd1 = np.asarray(atm.bhyd, dtype=np.float64)[:, 0]     # departure coeff ground level (=1 LTE)
xnfhe1 = np.asarray(atm.xnf_he1, dtype=np.float64)       # He I number density

tkev = K.KBOLTZ_EV * temp
tlog = np.log(temp)

# ── 2. CH / OH molecular populations from the equilibrium (taken as INPUT) ────
xnfpmol, _ = compute_mol_xnfpmol_dopple(atm, {246, 258}, molecules_path=MOL_DAT)
xnfpch = np.asarray(xnfpmol[246], dtype=np.float64)   # CH mode-1 population (N_CH/U_CH)
xnfpoh = np.asarray(xnfpmol[258], dtype=np.float64)   # OH mode-1 population (N_OH/U_OH)

# ── 3. Wavelength grid: span the optical/near-IR where H2-CIA dominates ───────
# Use a grid that covers visible -> near-IR (H2-CIA strongest ~1-2.5 um, but its
# tail extends through the optical for the densest cool layers).  CHOP needs E>2eV
# (<~6200 A); OHOP needs E>2.1eV; H2-CIA needs waveno<20000 cm^-1 (>500 nm).
# We pick a grid that hits all three species' active ranges.
wl_nm = np.linspace(300.0, 5000.0, 600)               # 300 nm .. 5 um
freq = K.C_LIGHT_NM / wl_nm                            # Hz
nfreq = freq.size

# ── 4. GROUND TRUTH: call pykurucz's own routines, per species ────────────────
stim = 1.0 - np.exp(-K.H_PLANCK * freq[None, :] / (K.K_BOLTZ * temp[:, None]))

chop_truth = np.zeros((n_layers, nfreq))
ohop_truth = np.zeros((n_layers, nfreq))
h2cia_truth = np.zeros((n_layers, nfreq))

for j in range(nfreq):
    f = freq[j]
    stim_j = stim[:, j]
    chop_truth[:, j] = K._chop_opacity(f, temp) * xnfpch / rho * stim_j
    ohop_truth[:, j] = K._ohop_opacity(f, temp) * xnfpoh / rho * stim_j
    h2cia_truth[:, j] = K._h2_collision_opacity(
        f, temp, xnfph1, bhyd1, xnfhe1, rho, tkev, tlog, stim_j
    )

mol_total_truth = chop_truth + ohop_truth + h2cia_truth

# ── 5. Save SHIPPED INPUTS (atmosphere state + equilibrium pops + tables) ─────
KT = K._KAPP_TABLES  # pykurucz kapp tables dict (CH/OH/H2 coefficient tables)
np.savez_compressed(
    REF / "mol_continuum_inputs.npz",
    # atmosphere state
    temperature=temp,
    mass_density=rho,
    xnfph1=xnfph1,
    bhyd1=bhyd1,
    xnfhe1=xnfhe1,
    # molecular equilibrium populations (taken as input)
    xnfpch=xnfpch,
    xnfpoh=xnfpoh,
    # wavelength grid
    wavelength_nm=wl_nm,
    frequency_hz=freq,
    # coefficient / Borysow tables (shipped self-contained)
    CH_PARTITION=KT["_CH_PARTITION"],
    OH_PARTITION=KT["_OH_PARTITION"],
    CH_CROSSSECT=KT["_CH_CROSSSECT"],
    OH_CROSSSECT=KT["_OH_CROSSSECT"],
    H2_COLL_H2H2=KT["_H2_COLL_H2H2"],
    H2_COLL_H2HE=KT["_H2_COLL_H2HE"],
)

# ── 6. Save GROUND TRUTH (for comparison only) ────────────────────────────────
np.savez_compressed(
    REF / "mol_continuum_truth.npz",
    chop=chop_truth,
    ohop=ohop_truth,
    h2cia=h2cia_truth,
    mol_total=mol_total_truth,
)

print("Wrote reference/mol_continuum_inputs.npz and reference/mol_continuum_truth.npz")
print(f"  n_layers={n_layers} nfreq={nfreq} wl=[{wl_nm.min():.0f},{wl_nm.max():.0f}] nm")
print(f"  CHOP   max = {chop_truth.max():.6e} cm^2/g")
print(f"  OHOP   max = {ohop_truth.max():.6e} cm^2/g")
print(f"  H2-CIA max = {h2cia_truth.max():.6e} cm^2/g")
# Dominance check at the densest (deepest) layer in the near-IR
jdeep = n_layers - 1
iIR = int(np.argmin(np.abs(wl_nm - 1200.0)))
print(f"  At deepest layer, 1.2um: CHOP={chop_truth[jdeep,iIR]:.3e} "
      f"OHOP={ohop_truth[jdeep,iIR]:.3e} H2CIA={h2cia_truth[jdeep,iIR]:.3e}")
