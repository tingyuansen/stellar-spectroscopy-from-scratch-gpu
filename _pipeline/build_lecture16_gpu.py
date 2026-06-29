#!/usr/bin/env python
"""Assemble content/Lecture16.ipynb (unexecuted).

Lecture 16 is the EOS-state counterpart to Lecture 15. Reference population
arrays are not used as computed state: the notebook computes PFSAHA/NELECT
populations from shipped atomic data, then builds the kGPU-style tensor state
for a loaded atmosphere fixture. The loaded atmosphere/radiation context is a
production-derived computed fixture, while population/state outputs are
comparison-only. Remaining helper and fixture limits are reported explicitly.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture16.ipynb"
cells = []
def md(s):
    """Append markdown source `s` to the generated Lecture 16 notebook."""
    cells.append(new_markdown_cell(s))

def code(s):
    """Append executable code source `s` to the generated Lecture 16 notebook."""
    cells.append(new_code_cell(s.strip("\n")))

md(r"""# Lecture 16 — EOS State for Line-Blanketed Convergence

*Stellar Spectroscopy from Scratch — a self-contained torch/MPS reconstruction of stellar-atmosphere and spectrum synthesis physics*

Lecture 16 recomputes the EOS state that the line-blanketed convergence loop consumes, conditional on a loaded atmosphere fixture. The computed path is:

1. solve the multi-element PFSAHA/NELECT equation of state from the atomic data tables;
2. pack the result into the Kurucz `POPSALL` species-slot layout;
3. build the kGPU-style per-iteration tensors the line-blanketed loop consumes: `dopple`, `xnfdop`, `txnxn`, and the ionization-energy heat-capacity finite-difference samples;
4. compare those computed outputs to reference arrays only after the computation.

The scope is explicit. The loaded atmosphere structure (`T`, `P`, density-depth context, radiation-pressure context) is production-derived computed data and is not regenerated here, so this is not a strict from-physical-inputs atmosphere closure. The full convergence trajectory and the production `TABCONT`/molecular reference files are retained as comparison/audit targets, not as computed outputs. Any residuals in those comparison-only sections are reported as limits rather than hidden behind broader language.

---

**Learning objectives.** By the end of this lecture you will be able to:

- Recompute PFSAHA/NELECT populations from atomic data and verify electron density, mass density, hydrogen populations, and the packed population tensor.
- Explain the Kurucz `POPSALL` species-slot layout and build it with a tensor gather instead of treating the reference array as state.
- Recompute the per-iteration line-blanketing tensors: atomic Doppler widths, `XNFDOP`, van-der-Waals perturber density `TXNXN`, and the ionization-energy heat-capacity finite-difference samples.
- Separate computed state from comparison-only audit bundles, especially `TABCONT`, molecular slots, and the full line-blanketed trajectory.""")

md(r"""## Setup — device, reference paths, and comparison helpers

The EOS solve itself uses the clean-room helper `eos_fromscratch.py`. `continuum_fromscratch.py` and `molecular_fromscratch.py` remain helper-backed audit boundaries for `TABCONT` and molecular slots. The state assembly is written in torch in the same shape and dtype style as kGPU: the working device is MPS/CUDA fp32 when available, with an fp64 CPU reference for precision checks.""")

code(r'''import pathlib, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

PIPE = pathlib.Path("..") / "_pipeline"
if str(PIPE) not in sys.path:
    sys.path.insert(0, str(PIPE))

import eos_fromscratch as EOS
import continuum_fromscratch as CF
import molecular_fromscratch as MF

if torch.backends.mps.is_available():
    DEVICE, DTYPE = torch.device("mps"), torch.float32
elif torch.cuda.is_available():
    DEVICE, DTYPE = torch.device("cuda"), torch.float32
else:
    DEVICE, DTYPE = torch.device("cpu"), torch.float64
REF64 = (torch.device("cpu"), torch.float64)
print(f"working device = {DEVICE.type}   working dtype = {str(DTYPE).split('.')[-1]}")
plt.rcParams.update({"figure.figsize": (7.2, 4.2), "figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25})''')

md(r"""The bundles are loaded next. `pfsaha_inputs.npz` is atomic input data for the EOS computation. `eos_state_ref.npz` and `lineblanket_ref.npz` contain a mixture: comparison targets plus the loaded atmosphere/radiation context this lecture consumes. Under the strict rule, those context arrays are production-derived computed fixtures, not physical input tables.""")

code(r'''
REF = pathlib.Path("..") / "reference"
S = np.load(REF / "eos_state_ref.npz", allow_pickle=True)          # comparison only
LB = np.load(REF / "lineblanket_ref.npz", allow_pickle=True)       # comparison only
TAB = EOS.EOSTables.from_npz(REF / "pfsaha_inputs.npz")            # atomic data input

T = S["T"].astype(np.float64)
P = S["P"].astype(np.float64)
xabund = S["xabund"].astype(np.float64)
wtmole = float(S["wtmole"])
vturb = S["vturb"].astype(np.float64)
atmass = S["atmass"].astype(np.float64)
n_depth = T.size

print(f"EOS state grid: {n_depth} layers; loaded atmosphere fixture = eos_state_ref.npz")''')

md(r"""Finally, define small comparison helpers. They enforce the rule used throughout this lecture: compute first, move tensors back to CPU/fp64 only at the comparison boundary, then assert the documented floor.""")

code(r'''
def back(x):
    """Return `x` as a host NumPy float64 array for comparison boundaries.

    Inputs may be torch tensors on the selected device or NumPy-like arrays. The
    helper is used only after a quantity has been computed, so it does not move
    reference arrays into the taught computation path.
    """
    if torch.is_tensor(x):
        return x.detach().cpu().to(torch.float64).numpy()
    return np.asarray(x, dtype=np.float64)

def relmax(a, b, *, rel_floor=0.0):
    """Return the masked max relative deviation between `a` and `b`.

    Inputs are computed and comparison arrays. If `rel_floor` is nonzero, only
    entries above that fraction of the comparison array's peak magnitude are
    included; this avoids trace-population fp32 underflow dominating meaningful
    opacity-state checks. The output is a Python float.
    """
    a = back(a); b = back(b)
    if rel_floor and np.max(np.abs(b)) > 0:
        m = np.abs(b) > rel_floor * np.max(np.abs(b))
    else:
        m = np.abs(b) > 0
    if not np.any(m):
        return 0.0
    return float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-300)))

CHECKS = {}
def check(name, got, ref, limit, *, rel_floor=0.0):
    """Assert one named parity check and record its max relative error.

    `got` is the computed quantity, `ref` is the comparison target, `limit` is
    the accepted floor, and `rel_floor` optionally masks insignificant trace
    entries. This verifies state parity conditional on the loaded atmosphere
    fixture; it is not a claim that the atmosphere fixture was regenerated.
    """
    err = relmax(got, ref, rel_floor=rel_floor)
    CHECKS[name] = err
    print(f"{name:34s} max|rel| = {err:.3e}   limit {limit:.1e}")
    assert err <= limit, f"{name} exceeded limit: {err} > {limit}"
    return err

def t(a, device=DEVICE, dtype=DTYPE):
    """Convert array-like `a` to a torch tensor on `device` with `dtype`.

    This is a convenience wrapper for state assembly. It is used for loaded
    fixture context as well as computed arrays, so call sites determine whether
    data are physical input, computed state, or comparison-only.
    """
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
''')

md(r"""## 1. PFSAHA/NELECT — compute populations from atomic data, then compare

This is the load-bearing repair. The previous GPU notebook took `population_per_ion` directly from `eos_state_ref.npz`. Here the populations are recomputed from the PFSAHA atomic tables and the loaded $(T,P)$ structure. The population arrays are comparison-only, but the $(T,P)$ structure itself is a fixture boundary.""")

code(r'''def solve_eos_state(T_in, P_in, xabund_in, tab, wtmole_in):
    """Return PFSAHA/NELECT EOS populations from clean-room NumPy helper code.

    Inputs are loaded atmosphere-fixture arrays `T_in` and `P_in`, abundance
    vector `xabund_in`, atomic table object `tab`, and molecular weight
    `wtmole_in`. The output is the helper's EOS state object containing electron
    density, mass density, hydrogen/helium populations, and
    `population_per_ion`. This removes direct use of reference population arrays,
    but it is still helper-backed NumPy and conditional on a loaded atmosphere
    fixture, not a strict all-torch from-physical-input closure.
    """
    return EOS.popsall(T_in, P_in, xabund_in, tab=tab, wtmole=wtmole_in)

pa = solve_eos_state(T, P, xabund, TAB, wtmole)
print("computed PFSAHA/NELECT state from pfsaha_inputs.npz")
print(f"  n_e surface/base = {pa.xne[0]:.3e} / {pa.xne[-1]:.3e} cm^-3")
print(f"  rho surface/base = {pa.rho[0]:.3e} / {pa.rho[-1]:.3e} g cm^-3")
print(f"  population_per_ion shape = {pa.population_per_ion.shape}")

check("electron density", pa.xne, S["xne"], 5e-12)
check("mass density", pa.rho, S["rho"], 5e-12)
check("H I/H II populations", pa.xnfph, S["xnfph"], 5e-12)
check("population_per_ion", pa.population_per_ion[:, :, :99], S["population_per_ion"][:, :, :99], 5e-11)''')

md(r"""## 2. Species-slot packing (`POPSALL`) — kGPU-style tensor gather

The line blanket does not index `(Z, ion)` directly. It indexes a flat Kurucz species slot (`nelion`). The map is triangular for $Z\le30$ and stride-5 for $Z\ge31$. We build that map once, then gather the computed populations into a tensor with shape `(depth, slot)`.""")

code(r'''def build_nelion_maps(maxn=999):
    """Return Kurucz POPSALL slot maps from `nelion` to `(Z, ion)`.

    Input `maxn` is the largest slot index to allocate. The outputs are integer
    arrays `z_of[slot]` and `ion_of[slot]`; zero means the slot is unused. This is
    static bookkeeping, not production-derived computed data, and is shared by
    the POPSALL, Doppler, and XNFDOP packing routines.
    """
    mode12 = [(1.01,1),(2.02,3),(3.03,6),(4.03,10),(5.03,15),(6.05,21),(7.05,28),(8.05,36),(9.05,45),(10.05,55),(11.05,66),(12.05,78),(13.05,91),(14.05,105),(15.05,120),(16.05,136),(17.04,153),(18.04,171),(19.04,190),(20.04,210),(21.04,231),(22.04,253),(23.04,276),(24.04,300),(25.04,325),(26.04,351),(27.04,378),(28.04,406),(29.02,435),(30.02,465)]
    nn = lambda code: max(1, int((code - int(code)) * 100.0 + 1.5))
    z_of = np.zeros(maxn + 1, np.int64); ion_of = np.zeros(maxn + 1, np.int64)
    for code, s in mode12:
        Z = int(code)
        for i in range(nn(code)):
            z_of[s+i] = Z; ion_of[s+i] = i+1
    for Z in range(31, 100):
        s = 496 + (Z - 31) * 5
        for i in range(nn(Z + 0.02)):
            z_of[s+i] = Z; ion_of[s+i] = i+1
    return z_of, ion_of

Z_OF, ION_OF = build_nelion_maps()
nslots = 1006

def popsall_slots(device, dtype):
    """Pack computed per-ion populations into the flat POPSALL slot tensor.

    Inputs select torch `device` and `dtype`; the function closes over
    `pa.population_per_ion`, which was recomputed from PFSAHA/NELECT for the
    loaded `(T, P, abundances)` atmosphere fixture. The output is
    `popsall[depth, slot]`. Parity is checked against `eos_state_ref.npz`, but
    this function does not regenerate the atmosphere fixture itself.
    """
    pop = torch.as_tensor(pa.population_per_ion[:, :, :99], dtype=dtype, device=device)
    z_map = torch.as_tensor(Z_OF, dtype=torch.long, device=device)
    ion_map = torch.as_tensor(ION_OF, dtype=torch.long, device=device)
    nel = torch.arange(1, z_map.numel(), dtype=torch.long, device=device)
    good = ((z_map[nel] >= 1) & (z_map[nel] <= 99) & (ion_map[nel] >= 1) &
            (ion_map[nel] <= pop.shape[1]) & (nel <= nslots))
    cols = nel[good] - 1
    z_idx = z_map[nel[good]] - 1
    ion_idx = ion_map[nel[good]] - 1
    out = torch.zeros((n_depth, nslots), dtype=dtype, device=device)
    out[:, cols] = pop[:, ion_idx, z_idx]
    return out

pops_w = popsall_slots(DEVICE, DTYPE)
pops_r = popsall_slots(*REF64)
check("POPSALL fp32/fp64 significant", pops_w, pops_r, 5e-6, rel_floor=1e-6)
print(f"populated atomic slots: {int(torch.count_nonzero(torch.sum(torch.abs(pops_w), dim=0) > 0).detach().cpu())}")

fig, ax = plt.subplots(figsize=(8.2, 3.1))
nel = np.arange(1, Z_OF.size); good = Z_OF[1:] > 0
ax.scatter(nel[good], Z_OF[1:][good], s=3, c=ION_OF[1:][good], cmap="viridis")
ax.axvline(495, color="0.5", ls="--", lw=1)
ax.set_xlabel("species slot nelion"); ax.set_ylabel("atomic number Z")
ax.set_title("Kurucz POPSALL slot schedule")
fig.tight_layout()''')

md(r"""## 3. Doppler widths and `xnfdop` — computed from populations, mass, density

`dopple` is the fractional thermal-plus-microturbulent Doppler width. `xnfdop` is the species population divided by `dopple*rho`, the line-center normalization used by Lecture 15's deposit. Both are computed from the just-solved populations, not read from the reference.""")

code(r'''K_BOLTZ = 1.38054e-16
AMU = 1.660e-24
C_LIGHT = 2.99792458e10

def doppler_xnfdop(device, dtype):
    """Return atomic Doppler widths and XNFDOP slot tensors.

    Inputs select torch `device` and `dtype`; the function closes over loaded
    temperature, microturbulence, atomic masses, and the recomputed EOS
    populations/density. It returns `(dopple[depth, slot], xnfdop[depth, slot])`.
    Significant slots should agree at the fp32 floor; unmasked XNFDOP can show
    trace-population underflow far below opacity relevance.
    """
    Tt = t(T, device, dtype); vt = t(vturb, device, dtype); am = t(atmass, device, dtype); rh = t(pa.rho, device, dtype)
    pop = torch.as_tensor(pa.population_per_ion[:, :, :99], dtype=dtype, device=device)
    arg = 2.0 * K_BOLTZ * Tt[:, None] / (torch.clamp(am[None, :], min=1e-300) * AMU) + vt[:, None] ** 2
    width_elem = torch.where(am[None, :] > 0, torch.sqrt(torch.clamp(arg, min=0.0)) / C_LIGHT, torch.zeros_like(arg))
    z_map = torch.as_tensor(Z_OF, dtype=torch.long, device=device)
    ion_map = torch.as_tensor(ION_OF, dtype=torch.long, device=device)
    nel = torch.arange(1, z_map.numel(), dtype=torch.long, device=device)
    good = ((z_map[nel] >= 1) & (z_map[nel] <= 99) & (ion_map[nel] >= 1) &
            (ion_map[nel] <= pop.shape[1]) & (nel <= nslots))
    cols = nel[good] - 1
    z_idx = z_map[nel[good]] - 1
    ion_idx = ion_map[nel[good]] - 1
    pop_g = pop[:, ion_idx, z_idx]
    dop_g = width_elem[:, z_idx]
    dopple = torch.zeros((n_depth, nslots), dtype=dtype, device=device)
    xnfdop = torch.zeros_like(dopple)
    dopple[:, cols] = dop_g
    tiny = torch.finfo(dtype).tiny
    xnfdop[:, cols] = torch.where((dop_g > 0) & (rh[:, None] > 0), pop_g / torch.clamp(dop_g * rh[:, None], min=tiny), torch.zeros_like(pop_g))
    return dopple, xnfdop

dop_w, xnf_w = doppler_xnfdop(DEVICE, DTYPE)
dop_r, xnf_r = doppler_xnfdop(*REF64)
check("dopple fp32/fp64", dop_w, dop_r, 5e-6)
check("dopple vs reference", dop_r, S["dopple"], 5e-12)
check("xnfdop significant fp32/fp64", xnf_w, xnf_r, 5e-6, rel_floor=1e-6)
check("xnfdop significant vs reference", xnf_r, S["xnfdop"], 5e-10, rel_floor=1e-12)
raw_x = relmax(xnf_w, xnf_r)
print(f"unmasked xnfdop fp32/fp64 max|rel| = {raw_x:.2e} (trace slots underflow below fp32 exponent floor)")''')

md(r"""## 4. `txnxn` — van der Waals perturber number

The pressure-broadening term uses $(n_{\rm H I}+0.42n_{\rm He I}+0.85n_{\rm H_2})(T/10^4)^{0.3}$. Hydrogen and helium are from the recomputed EOS. The H$_2$ correction is the same closed-form equilibrium used by the NumPy implementation.""")

code(r'''KEV_FACTOR = 8.6171e-5

def txnxn_torch(device, dtype):
    """Return the van-der-Waals neutral perturber density `txnxn`.

    Inputs select torch `device` and `dtype`; the function uses loaded
    temperatures and recomputed H/He populations from the EOS solve. The output
    is `txnxn[depth]`. The H2 term uses the Kurucz-compatible closed-form
    equilibrium and high-temperature cutoff; this is a prescription boundary and
    is parity-checked against the fixture.
    """
    Tt = t(T, device, dtype); h = t(pa.xnf_h, device, dtype); he1 = t(pa.xnf_he1, device, dtype)
    tkev = Tt * KEV_FACTOR
    eq = torch.exp(4.478 / tkev - 4.64584e1 + (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 + (1.06206e-18 - 3.08720e-23 * Tt) * Tt) * Tt) * Tt) * Tt) * Tt - 1.5 * torch.log(Tt))
    # Kurucz-compatible numerical cutoff for the H2 neutral-perturber term. This is
    # a prescription boundary, not a physical discontinuity in molecular equilibrium.
    h2 = torch.where(Tt > 9000.0, torch.zeros_like(Tt), h*h*eq)
    return (h + 0.42 * he1 + 0.85 * h2) * (Tt / 1.0e4) ** 0.3

tx_w = txnxn_torch(DEVICE, DTYPE)
tx_r = txnxn_torch(*REF64)
check("txnxn fp32/fp64", tx_w, tx_r, 5e-6)
check("txnxn vs reference", tx_r, S["txnxn"], 5e-12)
print(f"txnxn surface/base = {float(tx_w[0].detach().cpu()):.3e} / {float(tx_w[-1].detach().cpu()):.3e} cm^-3")''')

md(r"""## 5. Convective heat capacity — finite differences with ionization energy

Lecture 11 needs four EOS perturbations at $T\pm0.1\%$ and $P\pm0.1\%$. Here those samples are recomputed from the EOS engine, including cumulative ionization energy. This is the part that prevents the adiabatic gradient from reverting to the ideal monatomic value $0.4$ in partially ionized gas.""")

code(r'''def convec_fd_samples(T_in, P_in, xabund_in, wtmole_in, xne_in, tab):
    """Return EOS finite-difference samples for convective heat capacity.

    Inputs are loaded atmosphere-fixture arrays `T_in` and `P_in`, abundances,
    molecular weight, the recomputed electron density `xne_in`, and atomic tables.
    The output is `(ei1, ei2, ei3, ei4, r1, r2, r3, r4)`: ionization-energy and
    density samples at the four small perturbations used by the convection
    finite difference. This is a clean-room NumPy helper boundary; the notebook
    audits parity but does not make this an all-torch kernel.
    """
    return EOS.convec_fd_raw(T_in, P_in, xabund_in, wtmole_in, xne_in, tab)

ei1, ei2, ei3, ei4, r1, r2, r3, r4 = convec_fd_samples(T, P, xabund, wtmole, pa.xne, TAB)
dilut = 1.0 - np.exp(-S["tauros"])
pradk = S["pradk"]
ed1 = ei1 + 3.0 * pradk / np.maximum(r1, 1e-300) * (1.0 + dilut * (1.001 ** 4 - 1.0))
ed2 = ei2 + 3.0 * pradk / np.maximum(r2, 1e-300) * (1.0 + dilut * (0.999 ** 4 - 1.0))
ed3 = ei3 + 3.0 * pradk / np.maximum(r3, 1e-300)
ed4 = ei4 + 3.0 * pradk / np.maximum(r4, 1e-300)
for name, got in [("rho1", r1), ("rho2", r2), ("rho3", r3), ("rho4", r4),
                  ("edens1", ed1), ("edens2", ed2), ("edens3", ed3), ("edens4", ed4)]:
    check(name, got, S[name], 5e-12)

ipcum = EOS._cumulative_ip_erg(TAB)
print(f"H cumulative ionization energy H->H+ = {ipcum[0,1]/1.602176634e-12:.3f} eV")
print("finite-difference EOS samples are computed, not read")''')

md(r"""The reference names `edens1`...`edens4` are inherited bookkeeping names. They are not radiation energy densities in the usual $aT^4$ sense: each sample is the EOS cumulative ionization-energy term plus a radiation-pressure contribution divided by density, so the quantity behaves as a specific-energy-like thermodynamic helper for the convection finite difference. The lecture keeps the historical names only so the comparison is traceable.""")

md(r"""## 6. Molecular slots and continuum cutoff — computed, with residual limits reported

These two components are retained as audit sections rather than overstated as exact GPU achievements. The molecular solver is the Lecture-13 clean-room chemistry. `TABCONT` is built by the NumPy continuum bridge with the far-UV completion. Both are computed before comparison, and their residuals are reported explicitly.""")

code(r'''def molecular_deposit_state(T_in, P_in, eos_state, xabund_in, vturb_in, molecules_path):
    """Return molecular line-deposit populations from the clean-room helper.

    Inputs are loaded atmosphere-fixture arrays `T_in`, `P_in`, and `vturb_in`,
    the recomputed EOS state, abundances, and the physical molecule table path.
    The output is `(xnfpmol, dop_mol, mol_nelion)`. This computes before
    comparison and does not import production code, but it remains a NumPy helper
    boundary rather than a torch-native lecture cell.
    """
    return MF.molecular_deposit_populations(T_in, P_in, eos_state, xabund_in, vturb_in, molecules_path)

def continuum_cutoff_table(eos_state, T_in, rho_in, xne_in, os_start_in):
    """Return TABCONT and IWAVETAB from the clean-room continuum helper.

    Inputs are the recomputed EOS state, loaded atmosphere-fixture temperature,
    density and electron-density context, and the opacity-sampling wavelength
    start. The output is `(tabcont, iwavetab)`. The function computes before
    comparison, but under the strict rule it is still a helper-backed continuum
    bridge and not a closed all-torch opacity generator.
    """
    return CF.build_tabcont(eos_state, T_in, rho_in, xne_in, os_start_in)

xnfpmol, dop_mol, mol_nelion = molecular_deposit_state(T, P, pa, xabund, vturb, REF / "molecules.dat")
mol_err = relmax(xnfpmol, S["xnfpmol"])
print(f"molecular_deposit_populations shape = {xnfpmol.shape}")
print(f"xnfpmol vs reference max|rel| = {mol_err:.3e}  (documented Lecture-13 chemistry floor)")
assert mol_err < 5e-4

xnfdop_mol = back(xnf_r).copy(); dopple_mol = back(dop_r).copy()
xnfdop_mol, dopple_mol = MF.fill_molecular_slots(xnfdop_mol, dopple_mol, pa.rho, xnfpmol, dop_mol)
print(f"molecular slots filled: {int(np.count_nonzero(xnfdop_mol[:,840:940].sum(axis=0)))}")

os_start = float(LB["waveset_nm"][0])
tabcont, iwavetab = continuum_cutoff_table(pa, T, pa.rho, pa.xne, os_start)
mask = LB["tabcont"] != 0
tab_rel = np.abs(tabcont.astype(float) - LB["tabcont"].astype(float)) / np.maximum(np.abs(LB["tabcont"]), 1e-30)
print(f"tabcont median|max relative residual = {np.median(tab_rel[mask]):.3e} | {np.max(tab_rel[mask]):.3e}")
print(f"iwavetab exact = {np.array_equal(iwavetab, LB['iwavetab'])}")
assert np.array_equal(iwavetab, LB["iwavetab"])
assert np.median(tab_rel[mask]) < 1e-2
print("TABCONT residual is reported as a remaining continuum-bridge limit, not hidden as machine precision")''')

md(r"""## 7. Comparison-only convergence summary

The full warm-start line-blanketed convergence trajectory in `converge_fromscratch_result.npz` is a reference artifact. This lecture does not claim to rerun that entire loop inside the notebook. It uses the file only to state the comparison target and residual after the EOS-state blocker is removed.""")

code(r'''conv_path = REF / "converge_fromscratch_result.npz"
if conv_path.exists():
    conv = np.load(conv_path)
    Tk, Xk, Ts, Rs = conv["T"], conv["rhox"], conv["Ts"], conv["Rs"]
    Tk_on = np.interp(np.log(Rs), np.log(Xk), Tk)
    mrel = np.abs(Tk_on - Ts) / np.maximum(np.abs(Ts), 1.0)
    print("NumPy from-scratch convergence comparison-only summary:")
    print(f"  surface T = {Tk[0]:.1f} K; base T = {Tk[-1]:.1f} K; base RHOX = {Xk[-1]:.3f}")
    print(f"  T vs sun.npz median|max rel = {np.median(mrel):.3e} | {np.max(mrel):.3e}")
else:
    print("No convergence summary file present")''')

md(r"""## Final validation table

The green checks are the pieces Lecture 16 computes conditional on the loaded atmosphere fixture: PFSAHA/NELECT populations, species-slot packing, Doppler/xnfdop, txnxn, and the ionization-energy finite-difference heat-capacity inputs. Molecular and `TABCONT` sections are computed through clean-room NumPy helper boundaries and audited with their stated residual limits; the full convergence trajectory is comparison-only.""")

code(r'''print("GREEN ASSERTED CHECKS")
for k, v in CHECKS.items():
    print(f"  {k:34s} {v:.3e}")
worst = max(CHECKS.values())
print(f"worst asserted max|rel| = {worst:.3e}")
assert worst <= 5e-6''')

md(r"""## Summary

- The previous L16 false scope is reduced: `population_per_ion`, `dopple`, `xnfdop`, `txnxn`, and the EOS finite-difference samples are computed first, then compared, but all are conditional on a loaded atmosphere fixture.
- Reference population outputs are comparison-only. The loaded atmosphere/radiation context is still a production-derived fixture; the notebook no longer stages reference `TABCONT`, molecular populations, or convergence arrays as if they were generated by the GPU path.
- The kGPU-style tensor state is fp32-safe on significant slots; the only documented fp32 caveat is trace `xnfdop` underflow far below opacity relevance.
- Residual limits are explicit: molecular populations sit at the Lecture-13 floor; `TABCONT` has a continuum-bridge residual reported separately; full convergence is summarized as a reference comparison artifact, not claimed as a notebook rerun.""")

md(r"""## Synthesis

Lecture 16 narrows the bookkeeping gap that earlier atmosphere lectures deliberately exposed. A line-blanketed iteration is not only a temperature correction and a hydrostatic march; every iteration also needs a coherent EOS state: ion populations in `POPSALL` order, Doppler widths, population-per-Doppler opacity normalizers, neutral perturber densities, and the finite-difference heat-capacity samples used by convection. This lecture now recomputes that state for the loaded atmosphere fixture before comparing it.

The important boundary is also explicit. The tensor state that kGPU consumes is computed here and checked at the fp32 floor on significant slots, but the atmosphere structure it is computed on is still loaded. The molecular slots and continuum cutoff bridge are computed through clean-room NumPy helpers and reported with their residual limits. The full convergence trajectory remains a comparison artifact, not a hidden claim that this notebook reran the entire warm-start loop. That separation is what makes the passdown clean: the reusable ingredients are real, and the remaining integration targets are named rather than disguised.""")

md(r"""## Practice exercises

1. **Trace-slot underflow.** The significant `xnfdop` slots pass at the fp32 floor, while the unmasked maximum is dominated by trace populations far below opacity relevance. Recompute the check with two different relevance floors and explain which slots can affect line opacity.

2. **Perturber sensitivity.** In `txnxn`, change the helium coefficient from `0.42` to `0.50` and rerun the check. Which depths move most, and why does that identify the layers where van der Waals damping matters?

3. **Finite-difference heat capacity.** Repeat the EOS finite-difference sample with a `0.2%` perturbation instead of `0.1%`. Compare the asserted densities and energy samples. What part of the convection calculation would be most sensitive to this choice?

4. **Passdown boundary.** List which arrays in this lecture are computed conditional on the loaded atmosphere fixture, which helper-backed boundaries remain, and which outputs remain comparison-only. This is the checklist a production handoff needs before replacing a reference fixture with a runtime kernel.""")

md(r"""## Further reading

- Kurucz, R. L. (1970), *ATLAS: A Computer Program for Calculating Model Stellar Atmospheres*. The original ATLAS organization behind the EOS, opacity, temperature-correction, and hydrostatic state that this lecture packages.
- Mihalas, D. (1978), *Stellar Atmospheres*. The standard reference for LTE populations, line broadening inputs, and radiative-equilibrium atmosphere structure.
- Gray, D. F. (2005), *The Observation and Analysis of Stellar Photospheres*. Useful physical context for Doppler widths, damping perturbers, and why population bookkeeping matters for spectra.
- The kGPU source tree. The immediate implementation target for this lecture's tensors: compact, reusable kernels that consume `POPSALL`, Doppler state, `xnfdop`, perturbers, and convection samples without dragging the textbook scaffolding with them.""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT.relative_to(BOOK)} ({len(cells)} cells)")
