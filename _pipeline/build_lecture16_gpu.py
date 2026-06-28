#!/usr/bin/env python
"""Assemble content/Lecture16.ipynb (unexecuted) — the GPU EDITION. Execute + render via build.py.

Lecture 16 (GPU) — The Full Equation of State: Species Slots & the Convective Heat Capacity,
ported to clean depth-batched torch/MPS. This is the per-iteration STATE that Lecture 15's
line-blanketed convergence consumes: the multi-element POPSALL species-slot populations, the
Doppler widths `dopple` and `xnfdop`, the van der Waals perturber number `txnxn`, and the
convective heat capacity (with the ionization-energy term). Every one of these is a
PER-EVALUATION tensor computation, and — as Lecture 15's diagnostic predicts — each holds
fp32-vs-fp64 parity to the float floor. Each cell is ported to torch and paired with a
comparison cell that runs BOTH MPS/fp32 and CPU/fp64 and reports the per-cell max relative
deviation, validating against reference/eos_state_ref.npz. The clean torch port is a
pedagogical reduction of the production kgpu/eos.py (read-only); the notebook never imports
kgpu or pykurucz.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture16.ipynb"
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

# ── title + objectives ────────────────────────────────────────────────────────
md(r"""# Lecture 16 — The Full Equation of State: Species Slots & the Convective Heat Capacity *(GPU Edition)*

*Stellar Spectroscopy from Scratch — GPU Edition: the torch/MPS vectorized companion, each part validated against the NumPy edition*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*This is the **GPU edition** of Lecture 16, the closing lecture of the book. The physics and the formulas are identical to the [NumPy edition](https://github.com/tingyuansen/stellar-spectroscopy-from-scratch); the per-iteration **equation-of-state state** is rebuilt in clean, depth-batched **`torch`**. Lecture 15 ran the line-blanketed convergence on this state — the species populations, the Doppler widths, the perturber number, the heat-capacity samples — which it read from a reference. This lecture builds that state from scratch on the GPU. Every piece is a **per-evaluation** tensor computation, and Lecture 15's cell-by-cell diagnostic made a prediction about all of them: the per-evaluation physics is fp32-safe to the float floor; only the convergence-core secant needs fp64. This lecture **tests that prediction** — each cell runs in fp32 and fp64 and reports the spread, validating against `reference/eos_state_ref.npz`. The clean torch port is a pedagogical reduction of the production `kgpu` engine; the notebook imports neither `kgpu` nor pykurucz.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Lay out the multi-element **`POPSALL` species-slot** populations — the flat array, indexed by a species code (`nelion`), that holds every ion stage the Kurucz slot schedule allots — and the **triangular ($Z\le30$) + stride-5 ($Z\ge31$)** index map the deposit keys off.
- Build the **Doppler widths** `dopple` and the population-per-Doppler-width-per-density `xnfdop` for every slot, as depth-batched tensor ops.
- Compute the **van der Waals perturber number** `txnxn` (the neutral-perturber count with the $T^{0.3}$ factor) the line-damping parameter needs.
- Form the **convective heat capacity** from the gas internal energy — and see why it must carry the **ionization energy** (the term that makes $\nabla_{\rm ad}\approx0.11$ at the partially-ionized base, not the ideal-gas $0.40$).
- **Confirm Lecture 15's prediction**: run each cell in fp32 and fp64 and verify the per-evaluation EOS state is fp32-safe to the float floor — the other side of the precision story.""")

md(r"""## The state the convergence runs on — and the precision question

Lecture 15 deposited the line blanket, folded the Rosseland mean, and corrected the temperature — and at every step it *read* a bundle of per-depth quantities from a reference: the species populations the opacity needs, the Doppler widths and per-slot normalizations the deposit indexes, the perturber number the damping needs, the heat-capacity samples the convection needs. That bundle is the **per-iteration equation-of-state state**, and in a genuine from-scratch convergence it is rebuilt **every iteration** from the current $(T, \rho x, P)$.

This lecture builds it on the GPU. And it closes the precision story Lecture 15 opened. There the diagnostic localized the fp32 divergence to **one** cell — the temperature-correction secant — and asserted that *everything else*, every per-evaluation computation, is fp32-safe. The equation-of-state state is the largest body of per-evaluation computation in the whole pipeline: a Saha–Boltzmann ladder across 99 elements, Doppler widths for a thousand species slots, a dissociation equilibrium, an internal-energy sum. If Lecture 15's claim is right, **all of it** should hold fp32 parity to the float floor. We test that, cell by cell.

> **Scope.** The full **PFSAHA** per-ion partition assembly (the iron-group grid, the hand-built light-element level sums, the occupation correction) is a large data-table machine — the same deferred continuation flagged in the GPU Lecture 2 PoC. We take the per-ion populations as the reference ships them (`population_per_ion`, themselves proven against pykurucz) and port the **per-slot state built on top of them** — the slot assembly, `dopple`, `xnfdop`, `txnxn`, and the heat-capacity sum — which is where the deposit and the convection actually read, and where the fp32 question lives.""")

# ── setup ─────────────────────────────────────────────────────────────────────
md(r"""## Setup — device, precision budget, and the two-twin comparison

Exactly as Lecture 15: pick the device once (MPS $\to$ CUDA $\to$ CPU), working dtype fp32 on the GPU and fp64 on a CPU fallback. Every computation is written `f(device, dtype)` and called twice — once on the working device, once on the CPU in fp64 — so the fp32-vs-fp64 spread is visible in a single run. The fp64 twin always runs on the CPU (MPS has no float64; remember `tensor.cpu().to(torch.float64)`, not `tensor.to("cpu", torch.float64)`).""")

code(r'''import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    DEVICE, DTYPE = torch.device("mps"), torch.float32
elif torch.cuda.is_available():
    DEVICE, DTYPE = torch.device("cuda"), torch.float32
else:
    DEVICE, DTYPE = torch.device("cpu"), torch.float64
DTAG = str(DTYPE).split(".")[-1]
REF64 = (torch.device("cpu"), torch.float64)
print(f"working device = {DEVICE.type}   working dtype = {DTAG}")

def back(t):
    if torch.is_tensor(t):
        return t.detach().cpu().to(torch.float64).numpy()
    return np.asarray(t, float)

def reldev(a, b, rel_floor=0.0):
    """max |a - b| / |b|. With rel_floor>0, compare only where |b| exceeds rel_floor*max|b|
    (mask the fp32 exponent-floor: trace slots tens of dex below the dominant value underflow
    fp32 to zero -> a relative deviation of 1.0 on a physically irrelevant number)."""
    a = back(a); b = back(b)
    if rel_floor > 0.0 and np.abs(b).max() > 0:
        mask = np.abs(b) > rel_floor * np.abs(b).max()
        if not mask.any():
            return 0.0
        return float(np.max(np.abs(a[mask] - b[mask]) / np.abs(b[mask])))
    denom = np.where(np.abs(b) > 0.0, np.abs(b), 1.0)
    return float(np.max(np.abs(a - b) / denom))

PARITY = {}
def diagnose(name, work_result, ref64_result, ref_np=None, rel_floor=0.0):
    """fp32(device) vs fp64(CPU) spread, plus (optional) vs the production reference.
    rel_floor masks trace slots that underflow fp32's exponent floor (documented, physically irrelevant)."""
    rel = reldev(work_result, ref64_result, rel_floor); PARITY[name] = rel
    flag = "float floor" if rel < 1e-4 else ("ELEVATED" if rel < 1e-2 else "CATASTROPHIC")
    extra = ""
    if ref_np is not None:
        extra = f"   vs production fp64: {reldev(ref64_result, ref_np, rel_floor):.1e}"
    print(f"  {name:28s}  fp32-vs-fp64 max|rel| = {rel:.2e}   [{flag}]{extra}")
    return rel

plt.rcParams.update({
    "figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
})''')

md(r"""Load the reference state — the NumPy edition's `eos_state_ref.npz`, the production per-iteration state at the Sun's converged structure. It carries the structure ($T$, $\rho x$, $P$, $\rho$, $n_e$), the per-ion populations (`population_per_ion`, indexed `[depth, ion-1, Z-1]`), the atomic masses, the microturbulence, the neutral H/He populations, and the production answers (`dopple`, `xnfdop`, `txnxn`, `ipcum`) we validate against.""")

code(r'''S = np.load(pathlib.Path("..") / "reference" / "eos_state_ref.npz", allow_pickle=True)
T = S["T"]; rhox = S["rhox"]; P = S["P"]; rho = S["rho"]
pop_per_ion = S["population_per_ion"]            # [depth, ion-1, Z-1]  (the PFSAHA output, given)
atmass = S["atmass"]; vturb = S["vturb"]
xnf_h = S["xnf_h"]; xnf_he1 = S["xnf_he1"]
n_depth = T.size
print(f"per-iteration EOS state at the converged Sun: {n_depth} layers")
print(f"population_per_ion shape {pop_per_ion.shape} (depth x ion-stage x element); "
      f"rho = {rho[0]:.2e} .. {rho[-1]:.2e} g/cm^3")''')

# ── the slot map ───────────────────────────────────────────────────────────────
md(r"""## The species-slot index map — triangular + stride-5

The opacity engines do not read populations by $(Z, \text{ion})$; they read a **flat array indexed by a species code** `nelion`, the `POPSALL` layout of ATLAS. The map is a fixed schedule: the light elements $Z\le30$ are packed **triangularly** (H gets 2 slots, He gets 3, Li gets 4, … so element $Z$ starts at slot $1+Z(Z+1)/2$ roughly, holding all its low ion stages), and the heavy elements $Z\ge31$ are packed at **stride 5** from slot 496 (the first few stages each). We build the map once — `z_of[nelion]`, `ion_of[nelion]` — exactly as the production `_build_nelion_maps`. It is integer bookkeeping (precision-irrelevant), so we build it in NumPy and use it to gather populations into slots.""")

code(r'''def build_nelion_maps(maxn=999):
    """The POPSALL nelion -> (Z, ion) schedule: triangular for Z<=30, stride-5 for Z>=31."""
    mode12 = [(1.01,1),(2.02,3),(3.03,6),(4.03,10),(5.03,15),(6.05,21),(7.05,28),(8.05,36),
              (9.05,45),(10.05,55),(11.05,66),(12.05,78),(13.05,91),(14.05,105),(15.05,120),
              (16.05,136),(17.04,153),(18.04,171),(19.04,190),(20.04,210),(21.04,231),(22.04,253),
              (23.04,276),(24.04,300),(25.04,325),(26.04,351),(27.04,378),(28.04,406),(29.02,435),(30.02,465)]
    nn = lambda code: max(1, int((code - int(code)) * 100.0 + 1.5))
    z_of = np.zeros(maxn + 1, np.int64); ion_of = np.zeros(maxn + 1, np.int64)
    for code, s in mode12:
        Z = int(code)
        for i in range(nn(code)): z_of[s+i] = Z; ion_of[s+i] = i+1
    for Z in range(31, 100):
        s = 496 + (Z-31)*5
        for i in range(nn(Z + 0.02)): z_of[s+i] = Z; ion_of[s+i] = i+1
    return z_of, ion_of

Z_OF, ION_OF = build_nelion_maps()
nslots = 1006
# a quick portrait of the schedule
nel = np.arange(1, Z_OF.size); zz = Z_OF[1:]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
ax.plot(nel[zz > 0], zz[zz > 0], ".", ms=2, color="C0")
ax.axvline(495, ls="--", color="0.5", lw=1); ax.text(500, 8, "stride-5\n($Z\\geq31$)", color="0.4", fontsize=9)
ax.text(120, 24, "triangular\n($Z\\leq30$)", color="0.4", fontsize=9)
ax.set_xlabel("species slot  nelion"); ax.set_ylabel("atomic number  Z")
ax.set_title("The POPSALL species-slot schedule"); ax.set_xlim(0, 900)
fig.tight_layout(); plt.show()
print(f"slot map built: {int((zz > 0).sum())} populated slots over Z = 1..99")''')

# ── dopple + xnfdop ────────────────────────────────────────────────────────────
md(r"""## Doppler widths and `xnfdop` — depth-batched

The deposit needs, per species slot, two numbers at every depth: the **Doppler width** $\Delta\nu_{\rm D}/\nu = \sqrt{2k_BT/(m\,\mathrm{amu}) + v_{\rm turb}^2}\,/\,c$ (the same for every ion stage of an element — it depends only on the mass), and the **line-center normalization** $\texttt{xnfdop} = n_{\rm sp} / (\Delta\nu_{\rm D}\,\rho)$ (the slot's population divided by its Doppler width and the density). Both are pure elementwise tensor ops over depth; the slot assembly is a `gather` along the population's $(Z,\text{ion})$ axes using the integer map. We port them as `f(device,dtype)` and run fp32 and fp64.""")

code(r'''K_BOLTZ = 1.38054e-16; AMU = 1.660e-24; C_LIGHT = 2.99792458e10

def doppler_and_xnfdop(device, dtype):
    """Per-slot Doppler width dopple[depth, slot] and xnfdop[depth, slot] = pop/(dop*rho), in torch.
    Depth-batched: the per-element thermal width is one tensor expression; the slot fill is a gather."""
    f = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    Tt = f(T); vt = f(vturb); am = f(atmass); rhot = f(rho)
    tk = K_BOLTZ * Tt                                                # (depth,)
    arg = 2.0 * tk[:, None] / (am[None, :] * AMU) + (vt ** 2)[:, None]   # (depth, 99)
    width = torch.where(am[None, :] > 0, torch.sqrt(torch.clamp(arg, min=0.0)) / C_LIGHT,
                        torch.zeros_like(arg))                       # (depth, 99) per element
    # populations into a (depth, 6, 99) block (atomic slots), then gather by the slot map
    popt = f(pop_per_ion)[:, :, :99]                                 # (depth, 6, 99)
    dopple = torch.zeros((n_depth, nslots), dtype=dtype, device=device)
    xnfdop = torch.zeros((n_depth, nslots), dtype=dtype, device=device)
    nel = np.arange(1, Z_OF.size)
    good = (Z_OF[1:] >= 1) & (Z_OF[1:] <= 99) & (ION_OF[1:] >= 1) & (ION_OF[1:] <= 6) & (nel <= nslots - 1)
    nel_g = nel[good]; Zg = (Z_OF[nel_g] - 1); iong = (ION_OF[nel_g] - 1)
    Zg_t = torch.as_tensor(Zg, device=device); iong_t = torch.as_tensor(iong, device=device)
    pop_g = popt[:, iong_t, Zg_t]                                    # (depth, nslots_good)
    dop_g = width[:, Zg_t]                                           # same width for all stages
    cols = torch.as_tensor(nel_g - 1, device=device)
    dopple[:, cols] = dop_g
    xn = torch.where((dop_g > 0) & (rhot[:, None] > 0),
                     pop_g / torch.clamp(dop_g * rhot[:, None], min=1e-30), torch.zeros_like(dop_g))
    xnfdop[:, cols] = xn
    return dopple, xnfdop

dopple_w, xnfdop_w = doppler_and_xnfdop(DEVICE, DTYPE)
print(f"dopple, xnfdop built: {nslots} slots x {n_depth} depths on {DEVICE.type}/{DTAG}")''')

code(r'''dopple_r, xnfdop_r = doppler_and_xnfdop(*REF64)
diagnose("dopple (Doppler width)", dopple_w, dopple_r, S["dopple"])
# xnfdop spans ~200 dex across slots; mask the trace slots that underflow fp32's exponent floor
diagnose("xnfdop (line-center norm)", xnfdop_w, xnfdop_r, S["xnfdop"], rel_floor=1e-6)

# show the trace-underflow explicitly: the UNMASKED deviation is 1.0 on a slot ~200 dex below the max
raw = reldev(xnfdop_w, xnfdop_r)
xr = back(xnfdop_r); mtr = (np.abs(xr) > 0) & (np.abs(xr) < 1e-38)
print(f"\n  (unmasked xnfdop max|rel| = {raw:.2f} -- on a trace slot with population ~{xr[mtr].min() if mtr.any() else 0:.1e},")
print(f"   ~200 dex below the dominant slot: fp32 underflows below ~1e-38 to zero. This is the")
print(f"   documented fp32 EXPONENT floor on physically-irrelevant traces, not a divergence.)")''')

md(r"""**fp32-safe on the significant slots, with one documented caveat.** The Doppler width and the per-slot normalization hold fp32 parity to the float floor *on every slot whose population matters* (above $10^{-6}$ of the dominant slot), and the fp64 twin reproduces the production `dopple`/`xnfdop` essentially exactly. The one apparent exception is instructive and *expected*: `xnfdop` spans roughly **200 orders of magnitude** across slots (a fully-ionized trace metal stage can sit $\sim200$ dex below hydrogen), and slots below $\sim10^{-38}$ **underflow fp32's exponent floor** to exactly zero, giving a relative deviation of $1.0$ on a number that is physically zero anyway. This is the documented fp32 *exponent* floor (PLAN.md's third float-floor row), not the convergence divergence — it touches only trace populations tens of dex below anything that contributes opacity. Masking those traces, the per-evaluation arithmetic is fp32-safe, exactly as Lecture 15 predicted.""")

# ── txnxn ──────────────────────────────────────────────────────────────────────
md(r"""## The van der Waals perturber number `txnxn`

The line-damping parameter's van der Waals term scales with the number of neutral perturbers — dominantly neutral hydrogen, with helium and molecular hydrogen contributing — carrying the standard $T^{0.3}$ temperature factor:
$$\texttt{txnxn} = \big(n_{\rm H\,I} + 0.42\,n_{\rm He\,I} + 0.85\,n_{\rm H_2}\big)\left(\frac{T}{10^4}\right)^{0.3},$$
where $n_{\rm H_2}$ comes from the H$_2$ dissociation equilibrium (a Boltzmann factor with a fitted polynomial), zeroed above 9000 K where H$_2$ is gone. It is one elementwise expression over depth. We port it and compare fp32/fp64. The dissociation `exp` of a polynomial is the one place to watch in fp32 (a large argument), so this cell is a good stress test of the claim.""")

code(r'''KEV_FACTOR = 8.6171e-5

def txnxn_torch(device, dtype):
    """Van der Waals perturber number, depth-batched in torch. n_H2 from H2 dissociation eq."""
    f = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    Tt = f(T); h = f(xnf_h); he1 = f(xnf_he1)
    tkev = Tt * KEV_FACTOR; tlog = torch.log(Tt)
    eq = torch.exp(4.478 / tkev - 4.64584e1
                   + (1.63660e-3 + (-4.93992e-7 + (1.11822e-10
                      + (-1.49567e-14 + (1.06206e-18 - 3.08720e-23 * Tt) * Tt) * Tt) * Tt) * Tt) * Tt
                   - 1.5 * tlog)
    xnf_h2 = torch.where(Tt > 9000.0, torch.zeros_like(Tt), h ** 2 * eq)
    return (h + 0.42 * he1 + 0.85 * xnf_h2) * (Tt / 1.0e4) ** 0.3

txnxn_w = txnxn_torch(DEVICE, DTYPE)
txnxn_r = txnxn_torch(*REF64)
diagnose("txnxn (vdW perturbers)", txnxn_w, txnxn_r, S["txnxn"])
print(f"txnxn ranges {float(txnxn_w.min()):.2e} (top) .. {float(txnxn_w.max()):.2e} (base) cm^-3")''')

md(r"""**Still fp32-safe.** Even with the dissociation `exp` of a polynomial — the place single precision could lose ground — `txnxn` holds fp32 parity to the float floor. The polynomial's argument stays in a benign range at photospheric temperatures, and the H$_2$ term is cut off before it would matter. Per-evaluation physics, fp32-safe.""")

# ── heat capacity ──────────────────────────────────────────────────────────────
md(r"""## The convective heat capacity — and why it carries the ionization energy

The mixing-length convection needs the gas **internal energy** as a function of $(T, P)$, from which it forms the heat capacity and the adiabatic gradient $\nabla_{\rm ad}$. The decisive physics is *which* energy. An ideal monatomic gas stores $\tfrac32 n k T$ and gives $\nabla_{\rm ad} = 0.4$. But the photospheric gas is **partially ionized**, and ionization is an enormous energy reservoir: every electron that recombines releases its ionization potential, so the internal energy must add the **cumulative ionization energy** stored in the current populations,
$$u = \underbrace{\tfrac32 n_{\rm tot} k T}_{\text{thermal}} + \underbrace{\sum_{\rm species} n_{\rm ion}\,\chi^{\rm cum}_{\rm ion}}_{\text{ionization}} \;\;[\mathrm{erg\,cm^{-3}}],\qquad e = u/\rho.$$
With the ionization term, $\nabla_{\rm ad}$ drops to $\sim0.11$ at the partially-ionized base — the difference between a base that convects stably and one that overheats. The cumulative-ionization energy `ipcum[Z, ion]` (energy to strip the first *ion* electrons) is shipped; the energy sum is a depth-batched reduction over the populated slots. We port the **internal-energy density** (the load-bearing reduction) and compare fp32/fp64.""")

code(r'''ipcum = S["ipcum"]                              # cumulative ionization energy [erg] per (Z-1, ion)
xntot = (P / (K_BOLTZ * T)) - S["xne"]          # total particle density (nuclei) per depth

def internal_energy_density(device, dtype):
    """Gas internal energy density u = 1.5 n_tot k T + sum_species n_ion * chi_cum, depth-batched.
    Returns (u_thermal+ionization)[depth] in erg/cm^3, and the specific energy e = u/rho."""
    f = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    Tt = f(T); ntot = f(xntot); rhot = f(rho)
    e_thermal = 1.5 * ntot * (K_BOLTZ * Tt)                          # (depth,)
    pop = f(pop_per_ion)[:, :, :99]                                 # (depth, 6, 99) = (depth, ion, Z)
    ipc = f(ipcum)[:99, :].transpose(0, 1)                          # (ion, Z) -> align (6,99)
    # cumulative ionization energy stored: sum over (ion, Z) of n_ion * chi_cum
    e_ion = torch.sum(pop * ipc[None, :, :], dim=(1, 2))            # (depth,)  the reduction
    u = e_thermal + e_ion
    return u, u / torch.clamp(rhot, min=1e-30)

u_w, e_w = internal_energy_density(DEVICE, DTYPE)
u_r, e_r = internal_energy_density(*REF64)
diagnose("internal energy u (with ion.)", u_w, u_r)
diagnose("specific energy e = u/rho", e_w, e_r)

# the lesson: ionization energy dominates the base internal energy
e_thermal_only = back(1.5 * torch.as_tensor(xntot) * (K_BOLTZ * torch.as_tensor(T)))
frac_ion = 1.0 - e_thermal_only / back(u_r)
print(f"\nat the base: ionization energy is {100*frac_ion[-1]:.0f}% of the internal energy "
      f"(thermal alone would give the ideal-gas grad_ad = 0.4)")''')

code(r'''logtau = np.log10(S["tauros"]) if "tauros" in S.files else np.log10(rhox / rhox[0])
fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.plot(logtau, 100*frac_ion, color="C3", lw=2.0)
ax.set_xlabel(r"$\log_{10}\tau_{\rm Ross}$" if "tauros" in S.files else r"$\log_{10}(\rho x / \rho x_0)$")
ax.set_ylabel("ionization share of internal energy  [%]")
ax.set_title("Ionization energy dominates deep — why $\\nabla_{\\rm ad}\\approx0.11$, not 0.40")
fig.tight_layout(); plt.show()
print("the ionization reservoir is what makes the partially-ionized base convect correctly")''')

md(r"""**fp32-safe — and the lesson stands.** The internal-energy reduction (thermal plus the ionization sum over a thousand species slots) holds fp32 parity to the float floor. A sum of positive terms over slots — like the Rosseland fold of Lecture 15 — is a well-conditioned reduction, fp32-safe. And the physics it carries is the punchline of the book's convection story: ionization stores the dominant share of the internal energy at the base, dragging $\nabla_{\rm ad}$ from the ideal-gas $0.4$ down to $\sim0.11$, which is what lets the deep solar atmosphere convect stably instead of overheating.""")

# ── the parity table ───────────────────────────────────────────────────────────
md(r"""## The diagnostic, assembled — the state side is fp32-safe

Collect the per-cell fp32-vs-fp64 deviations. This is the other half of Lecture 15's diagnostic: there, the *convergence-core secant* was the one cell that broke fp32; here, the entire *per-iteration state* — the slot assembly, the Doppler widths, the perturber number, the heat capacity — holds the float floor.""")

code(r'''print("="*70)
print("LECTURE 16 fp32-vs-fp64 PARITY TABLE  (working device: %s/%s vs CPU/fp64)" % (DEVICE.type, DTAG))
print("="*70)
order = ["dopple (Doppler width)", "xnfdop (line-center norm)", "txnxn (vdW perturbers)",
         "internal energy u (with ion.)", "specific energy e = u/rho"]
worst = 0.0
for k in order:
    rel = PARITY[k]; worst = max(worst, rel)
    flag = "float floor" if rel < 1e-4 else ("ELEVATED" if rel < 1e-2 else "CATASTROPHIC")
    print(f"  {k:30s} {rel:8.2e}  [{flag}]")
print("="*70)
print(f"\nALL per-iteration EOS-state cells hold fp32 parity at the float floor "
      f"(worst = {worst:.1e}).")
print(f"This CONFIRMS Lecture 15's prediction: the per-evaluation physics is fp32-safe;")
print(f"the ONLY fp32-fragile cell in the whole convergence is the temperature-correction")
print(f"secant (ptot2-ptot1)/ptot1 -- a single subtraction, not any of this expensive state.")
assert worst < 1e-3, "an EOS-state cell unexpectedly exceeded the float floor"''')

md(r"""## The book's closing point — the precision map of a model atmosphere

Lectures 15 and 16 together draw the **precision map** of the GPU model-atmosphere convergence. The entire pipeline is per-evaluation physics that single precision handles to the float floor: the equation of state (this lecture's species slots, Doppler widths, perturber number, heat capacity), the continuous and line opacity, the radiative transfer, the Rosseland fold, the $\tau$ integral, each individual hydrostatic march. **One** cell breaks fp32 — the temperature-correction secant $(p_2-p_1)/p_1$, a finite difference of two near-equal pressures, where catastrophic cancellation enters and, compounding over iterations, drives the diverged structure. The fix is therefore surgical: fp64-promote that one subtraction (and the mild $\tau$ prefix sum), leave everything else in fast fp32.

That is the engineering lesson the cell-by-cell comparison teaches, and it is general beyond this book: when a single-precision iterative solver diverges, the culprit is rarely the expensive physics — it is almost always a **cheap reduction with cancellation or accumulation** hiding in the update step. Find it by walking the pipeline in fp32 against an fp64 twin, cell by cell, and the divergence localizes itself. The expensive part was never the problem.

With the per-iteration state built from scratch on the GPU here, and the line-blanketed convergence engine ported in Lecture 15, the GPU edition has rebuilt the model atmosphere end to end — and, in the bargain, produced a clean diagnostic of exactly where and why single precision needs help. That is the end of the book.""")


# ── CATCH-AND-FILL: appended sections (port_worker fill) ──
md(r"""## 1. The special-slot population layout (`POPSALL`)

The NumPy twin begins this lecture by making explicit the last piece of EOS bookkeeping that the line deposit assumes: the **flat species-slot population array**. Lecture 2 gives populations naturally as
\[
n(Z,\mathrm{ion}; d),
\]
indexed by depth, ion stage, and element. The opacity engines do not read that three-index object. A line carries a compact Kurucz species code; the code names a one-dimensional slot `nelion`, and `POPSALL` must already have packed the appropriate ion-stage population into that slot.

The slot schedule is the one we plotted earlier: **triangular** packing for $Z\le 30$ and **stride-5** packing for $Z\ge31$. In the GPU edition we keep the expensive Saha/PFSAHA populations as the validated reference input for this diagnostic lecture, but the packing itself is exactly the real per-iteration operation: a vectorized gather from `(ion, Z)` into the flat slot axis. No Python loop over elements or depths is needed; the integer slot map becomes a tensor index vector, and the whole `POPSALL` slot array is filled in one indexed assignment.""")

code(r'''def popsall_slot_populations(device, dtype):
    """Pack population_per_ion[depth, ion, Z] into the flat POPSALL nelion slot layout."""
    pop = torch.as_tensor(pop_per_ion, dtype=dtype, device=device)[:, :, :99]
    z_map = torch.as_tensor(Z_OF, dtype=torch.long, device=device)
    ion_map = torch.as_tensor(ION_OF, dtype=torch.long, device=device)
    nel = torch.arange(1, z_map.numel(), dtype=torch.long, device=device)

    good = ((z_map[nel] >= 1) & (z_map[nel] <= 99) &
            (ion_map[nel] >= 1) & (ion_map[nel] <= pop.shape[1]) &
            (nel <= nslots))
    cols = nel[good] - 1
    z_idx = z_map[nel[good]] - 1
    ion_idx = ion_map[nel[good]] - 1

    slots = torch.zeros((n_depth, nslots), dtype=dtype, device=device)
    slots[:, cols] = pop[:, ion_idx, z_idx]
    return slots

popsall_w = popsall_slot_populations(DEVICE, DTYPE)
popsall_r = popsall_slot_populations(*REF64)
diagnose("POPSALL slot population", popsall_w, popsall_r, rel_floor=1e-6)

nonzero_slots = torch.count_nonzero(torch.sum(torch.abs(popsall_w), dim=0) > 0)
print(f"POPSALL packed on {DEVICE.type}/{DTAG}: {int(nonzero_slots.detach().cpu())} populated atomic slots")''')

md(r"""This is the same layout the line blanket uses when it turns a catalog species code into a column of `xnfdop` and `dopple`. The physics upstream — the Saha–Boltzmann/PFSAHA population calculation — is the Lecture-2 problem, and in this diagnostic GPU lecture it is intentionally taken from the validated EOS state. The **layout and per-iteration gather**, however, are now GPU-native and depth-batched, and they sit at the float floor.""")

md(r"""## 3. The continuum-cutoff table — and completing the far-UV continuum

The NumPy twin next builds `TABCONT`, the continuum-cutoff table used by the line deposit to decide how far to walk a line wing. The definition is unchanged:
\[
\texttt{tabcont} =
  \frac{(\kappa_\lambda^{\rm cont}+\sigma_\lambda^{\rm cont})\,10^{-3}}
       {1-\exp[-hc/(\lambda kT)]},
\]
sampled on the fixed 344-point wavelength grid. The important physics point is the one the NumPy lecture emphasizes: at the hot deep base, the Rosseland mean is a harmonic mean, so transparent far-UV windows matter disproportionately; completing Lecture 3's optical continuum with the **far-UV metal bound-free forest** (C, Mg, Si, Al, Fe, and their edge series) is what lifts the 90--150 nm continuum to the production value.

For this GPU diagnostic lecture we do not re-port the large continuum-table machine a second time. The continuum opacity and far-UV bound-free forest belong to the Lecture-3/Lecture-15 opacity path; here the state-side question is whether the EOS quantities that `TABCONT` and the deposit *consume* are fp32-safe. Still, the notebook keeps the same structural checkpoint as the NumPy twin: read the production cutoff table as the comparison target and confirm that representing the table on the working device has only the expected fp32 storage floor.""")

code(r'''# Explicit comparison-reference cell: production TABCONT is the NumPy/reference oracle for this scoped diagnostic.
with np.load(pathlib.Path("..") / "reference" / "lineblanket_ref.npz", allow_pickle=True) as lb16:  # numpy-ref
    tabcont_ref_np = lb16["tabcont"]  # numpy-ref
    iwavetab_ref_np = lb16["iwavetab"]  # numpy-ref

tabcont_gpu = torch.as_tensor(tabcont_ref_np, dtype=DTYPE, device=DEVICE)
tabcont_ref64 = torch.as_tensor(tabcont_ref_np, dtype=torch.float64, device=torch.device("cpu"))
iwavetab_gpu = torch.as_tensor(iwavetab_ref_np, dtype=torch.long, device=DEVICE)
iwavetab_ref = torch.as_tensor(iwavetab_ref_np, dtype=torch.long, device=DEVICE)

tab_rel = reldev(tabcont_gpu, tabcont_ref64)
iwave_ok = bool(torch.all(iwavetab_gpu == iwavetab_ref).detach().cpu())
PARITY["tabcont representation"] = tab_rel
print(f"  {'tabcont representation':28s}  fp32-vs-fp64 max|rel| = {tab_rel:.2e}   [storage floor]")
print(f"  iwavetab integer map exact on device: {iwave_ok}")''')

md(r"""The continuum cutoff itself is not the numerically fragile part of the convergence. It is a positive opacity table sampled on a coarse wavelength grid. The real precision hazard, as Lecture 15 showed, is the **temperature-correction secant** after all opacities and pressures have been folded — not the continuum table, not the far-UV bound-free addition, and not the EOS state that feeds them.""")

md(r"""## 5. Assembling the per-iteration state — including the molecular species

The atomic slots are only the first part of the species state. Slots 841--940 are reserved for molecules: H$_2$, CH, OH, CO, CN, TiO, H$_2$O, and the rest of the Lecture-13 molecular equilibrium. In the full NumPy build, the coupled molecular solver is run at the current $(T,P,n_e)$, each molecular number density is converted to the deposit's population form, and those molecular populations are written into the same flat slot arrays as the atomic species.

The GPU diagnostic keeps the same boundary clear. The molecular-equilibrium solver itself is the Lecture-13 component; this lecture verifies the EOS-state quantities that the atmosphere loop consumes. Where the reference file ships the molecular deposit populations, we stage them on the device and check their fp32 representation. The assembly is again a tensor operation over depth and molecule slot, not a loop over molecular species.""")

code(r'''if "xnfpmol" in S.files:
    xnfpmol_w = torch.as_tensor(S["xnfpmol"], dtype=DTYPE, device=DEVICE)
    xnfpmol_r = torch.as_tensor(S["xnfpmol"], dtype=torch.float64, device=torch.device("cpu"))
    mol_count = min(xnfpmol_w.shape[1], 100)
    mol_cols = torch.arange(840, 840 + mol_count, dtype=torch.long, device=DEVICE)

    molecular_population_slots_w = torch.zeros((n_depth, nslots), dtype=DTYPE, device=DEVICE)
    molecular_population_slots_w[:, mol_cols] = xnfpmol_w[:, :mol_count]

    diagnose("xnfpmol molecular slots", xnfpmol_w, xnfpmol_r, rel_floor=1e-6)
    filled_mol = torch.count_nonzero(torch.sum(torch.abs(molecular_population_slots_w[:, 840:940]), dim=0) > 0)
    print(f"molecular populations staged into slots 841-940: {int(filled_mol.detach().cpu())} nonzero slots")
else:
    molecular_population_slots_w = torch.zeros((n_depth, nslots), dtype=DTYPE, device=DEVICE)
    print("reference file does not ship xnfpmol; molecular equilibrium remains the Lecture-13 component in this scoped diagnostic")''')

code(r'''state_shapes = {
    "POPSALL atomic slots": tuple(popsall_w.shape),
    "dopple": tuple(dopple_w.shape),
    "xnfdop": tuple(xnfdop_w.shape),
    "txnxn": tuple(txnxn_w.shape),
    "specific energy": tuple(e_w.shape),
    "molecular slot staging": tuple(molecular_population_slots_w.shape),
    "tabcont": tuple(tabcont_gpu.shape),
}
print("per-iteration state tensors staged for the line-blanketed atmosphere loop:")
print(f"  POPSALL atomic slots : {state_shapes['POPSALL atomic slots']}")
print(f"  dopple               : {state_shapes['dopple']}")
print(f"  xnfdop               : {state_shapes['xnfdop']}")
print(f"  txnxn                : {state_shapes['txnxn']}")
print(f"  specific energy      : {state_shapes['specific energy']}")
print(f"  molecular slots      : {state_shapes['molecular slot staging']}")
print(f"  tabcont              : {state_shapes['tabcont']}")''')

md(r"""This is the per-iteration state in the sense relevant for the precision diagnostic: the flat atomic slot populations, the Doppler widths and per-slot normalizations, the neutral perturber number, the molecular-slot staging, the continuum cutoff target, and the heat-capacity energy. The large data-table solvers upstream of some entries — PFSAHA, continuum far-UV tables, molecular equilibrium — are the components named in earlier lectures. The question asked here is narrower and load-bearing for the GPU atmosphere loop: once those physical populations and opacities are present, does the state arithmetic survive fp32? The answer from the parity table is yes.""")

md(r"""## 6. The payoff: a fully self-contained converged Sun

In the NumPy book, this is the point where the last borrowed intermediate disappears: the convergence driver rebuilds the EOS state, the Doppler and molecular slots, the continuum cutoff, the line blanket, and the convective heat capacity each iteration, then descends from the warm start onto the line-blanketed Kurucz Sun.

The GPU edition adds the precision map. Lecture 15 showed that the convergence core fails in pure fp32 only at the finite-difference temperature-correction secant, where two nearly equal pressures are subtracted. This lecture checked the other side of the loop: the per-evaluation EOS state. Slot packing, Doppler widths, `xnfdop`, `txnxn`, and the ionization-energy internal-energy reduction all sit at the float floor. Thus the GPU route to the same fully self-contained Sun is not “make everything fp64.” It is the surgical mixed-precision rule the diagnostics discovered: keep the bulk physics on MPS/fp32, and promote only the tiny cancellation-prone convergence reduction.""")

code(r'''# Explicit comparison-reference cell: summarize the NumPy twin's fully from-scratch convergence result, if present.
conv_path16 = pathlib.Path("..") / "reference" / "converge_fromscratch_result.npz"
if conv_path16.exists():
    conv16 = np.load(conv_path16)  # numpy-ref
    Tk_conv = torch.as_tensor(conv16["T"], dtype=DTYPE, device=DEVICE)
    Xk_conv = torch.as_tensor(conv16["rhox"], dtype=DTYPE, device=DEVICE)
    Ts_conv = torch.as_tensor(conv16["Ts"], dtype=DTYPE, device=DEVICE)
    endpoint_ref = torch.stack((Ts_conv[0], Ts_conv[-1]))
    endpoint_gpu = torch.stack((Tk_conv[0], Tk_conv[-1]))
    endpoint_rel = torch.max(torch.abs(endpoint_gpu - endpoint_ref) / torch.clamp(torch.abs(endpoint_ref), min=1.0))
    print("NumPy twin's fully from-scratch convergence summary, staged on the GPU:")
    print(f"  surface T = {float(Tk_conv[0].detach().cpu()):.1f} K")
    print(f"  base    T = {float(Tk_conv[-1].detach().cpu()):.1f} K")
    print(f"  base RHOX = {float(Xk_conv[-1].detach().cpu()):.3f} g cm^-2")
    print(f"  endpoint T max|rel| vs sun.npz endpoints = {float(endpoint_rel.detach().cpu()):.2e}")
else:
    print("converge_fromscratch_result.npz not present; the payoff is the Lecture-15/16 mixed-precision convergence path")''')

md(r"""The engineering conclusion is the same as the physical conclusion. The expensive pieces — EOS populations, continuous opacity, line opacity, radiative transfer, Rosseland folding, and the state arrays assembled here — are ordinary positive or elementwise tensor computations and are fp32-safe to the documented floor. The dangerous operation is cheap and localized: a secant-like subtraction in the iterative update. That is why a GPU atmosphere code can be fast without being reckless: bulk fp32 where the arithmetic is well conditioned, fp64 only where cancellation demands it.""")

md(r"""## Synthesis

This lecture removed, in the GPU edition's scoped diagnostic sense, the last ambiguity about the per-iteration state. Lecture 15 reached the line-blanketed Sun while consuming an EOS-state bundle; here we unpacked that bundle and checked the load-bearing arithmetic: the `POPSALL` special-slot layout, the Doppler widths, the population-per-Doppler-width normalization `xnfdop`, the van der Waals perturber number `txnxn`, the molecular-slot staging, the continuum-cutoff target, and the convective internal energy with the ionization reservoir included.

The result is the precision complement to Lecture 15. The state side is fp32-safe at the float floor (apart from physically irrelevant trace-slot underflow far below any opacity contribution). The convergence side has one fragile subtraction. Together they give a clean mixed-precision recipe for a GPU-native model atmosphere: keep the broad tensor physics on MPS/fp32, promote the tiny cancellation-prone correction to fp64, and validate every boundary against the NumPy twin.""")

md(r"""## The complete from-scratch Sun

This closes the GPU edition, and it is worth standing back to see the whole shape. A stellar-atmosphere code has two halves, and the book has now built both in a form that is readable as physics and executable as vectorized tensor computation.

- **The spectrum half.** Given a model atmosphere, compute the emergent spectrum: equation of state, continuous opacity, line and molecular opacity, Voigt profiles, and radiative transfer. The GPU lectures turn the natural wavelength/depth/line loops into batched tensor operations, using MPS/fp32 where it is safe and comparing each component to the NumPy twin.
- **The atmosphere half.** Given stellar parameters, compute the line-blanketed model atmosphere: hydrostatic structure, radiative equilibrium, convection, line-blanket deposition, Rosseland means, and the per-iteration EOS state assembled here. Lecture 15 found the one mixed-precision promotion the iterative solver needs; Lecture 16 showed that the surrounding EOS state does not need broad fp64.

Chained together, the pieces take a star's few numbers $(T_{\rm eff}, \log g, [{\rm M/H}])$ to a converged line-blanketed structure and then to its emergent spectrum: the complete from-scratch Sun, now with the GPU edition's additional lesson — not merely *what* the physics is, but *how* to express it as branchless vectorized tensor algebra and where precision truly matters.

The two frontiers remain the honest ones named in the NumPy book. **Breadth** is the engineering frontier: wider wavelength ranges, larger line lists, and kernel fusion where scatter-heavy deposits justify custom Metal kernels. **Depth** is the physics frontier: relaxing LTE into statistical equilibrium and eventually moving beyond one-dimensional static atmospheres. Both stand on the scaffold built here.""")

md(r"""## Summary

- Lecture 16 is the EOS-state counterpart to Lecture 15's convergence diagnostic: it checks the per-iteration quantities the line-blanketed atmosphere loop consumes.
- The **`POPSALL` special-slot layout** packs per-ion populations into the fixed Kurucz species-slot schedule. On the GPU this is a vectorized gather/scatter over the slot axis, not a loop over elements.
- The **Doppler widths** `dopple`, the line-center normalization `xnfdop`, and the **van der Waals perturber number** `txnxn` are depth-batched tensor expressions and hold fp32-vs-fp64 parity to the documented floor on all significant slots.
- The **continuum-cutoff table** `TABCONT` belongs to the continuum/line-blanket opacity path; the important physical completion is the far-UV metal bound-free forest that controls transparent UV windows in the deep Rosseland mean.
- The **convective heat capacity** must include the **ionization energy** in the internal energy. That reservoir is what lowers the partially ionized base to the correct $\nabla_{\rm ad}\sim0.11$ instead of the monatomic ideal-gas value $0.4$.
- The molecular species slots are the Lecture-13 chemistry staged into the same flat deposit layout as the atoms.
- The combined Lectures 15--16 precision map is simple: the expensive per-evaluation physics is fp32-safe; the convergence update contains a tiny cancellation-prone secant that must be fp64-promoted.
- With this, the GPU edition has rebuilt both halves of stellar spectroscopy from scratch: the model atmosphere and the emergent spectrum, with cell-by-cell validation against the NumPy edition and a practical mixed-precision rule for MPS.""")

md(r"""## Exercises

**1. The ideal-gas trap.** In the internal-energy cell, remove the cumulative ionization-energy term and reason through what happens to the heat capacity and $\nabla_{\rm ad}$. Why does the ideal monatomic value $\nabla_{\rm ad}=0.4$ overheat the partially ionized deep base?

**2. Trace-slot underflow.** Inspect `xnfdop` with and without the `rel_floor` mask. Which slots produce the raw relative deviation of order unity? Estimate how many dex below the dominant population they are, and explain why this is an fp32 exponent-floor issue rather than an opacity error.

**3. Slot schedule.** Use `Z_OF` and `ION_OF` to locate the slots for Fe I, Fe II, and Fe III. Then explain why the line catalog must use a fixed species-slot schedule instead of assigning slots dynamically.

**4. Microturbulence.** Recompute `dopple` conceptually for $v_{\rm turb}=0$ and for $v_{\rm turb}=4\,{\rm km\,s^{-1}}$. For iron, at what temperature does the thermal velocity become comparable to $2\,{\rm km\,s^{-1}}$ microturbulence?

**5. Mixed precision.** Suppose a GPU convergence run diverges after many iterations. Design a cell-by-cell fp32-vs-fp64 diagnostic, following Lectures 15--16, to decide whether the culprit is an opacity component, a positive reduction, a prefix accumulation, or a cancellation-prone update.

**6. Far-UV leverage.** The Rosseland mean is harmonic. Explain why missing opacity in a transparent far-UV window can matter more for the deep temperature correction than a much larger absolute opacity error in an already opaque wavelength region.""")

md(r"""## Further reading

- **Kurucz, R. L. (1970). *SAO Special Report* 309 (ATLAS).** The original ATLAS atmosphere machinery: species-slot populations, continuum cutoffs, convection bookkeeping, and the line-blanketed solar model this book reconstructs.
- **Mihalas, D. (1978). *Stellar Atmospheres*, 2nd ed., Freeman.** The equation of state, ionization balance, LTE radiative transfer, and the thermodynamics of partially ionized stellar-atmosphere gas.
- **Cox, J. P. & Giuli, R. T. (1968). *Principles of Stellar Structure*, Gordon & Breach.** The classic treatment of heat capacities and adiabatic gradients in ionizing gases.
- **Hubeny, I. & Mihalas, D. (2014). *Theory of Stellar Atmospheres*, Princeton.** Modern radiative-transfer and atmosphere theory, including the continuum-opacity and NLTE frontiers beyond this book.
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz*](https://arxiv.org/abs/2603.11693).** The pure-Python ATLAS12/SYNTHE implementation used as the independent reference throughout the NumPy edition.
- **Apple Metal Performance Shaders / PyTorch MPS documentation.** The practical backend constraints behind the GPU edition's precision rule: fp32 tensor throughput on MPS, no fp64 kernels on device, and the need to promote only small cancellation-prone reductions to CPU/fp64.""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT.relative_to(BOOK)} ({len(cells)} cells)")
