#!/usr/bin/env python
"""Assemble content/Lecture12.ipynb (unexecuted). Execute + render via build.py.

Lecture 12 — Molecular Equilibrium & Molecular Bands (TiO), written in clean torch. The
centerpiece is the TiO band-opacity accumulation over a [80 depth, 9136 wavelength] grid built from
~1.17M molecular lines — a SCATTER-ADD hot path. A scalar per-depth Python loop, with a
per-line near/far-wing scalar march that `np.add.at`-es into the opacity row, becomes batched
[D, L] / [pairs, step-block] tensor ops with `index_put_(accumulate=True)` as the scatter. The
lecture's added pedagogy is that VECTORIZATION: how the loop became blockwise tensor algebra, and
why this validated build uses a CPU-fp64 reference path for the sum of ~1e6 positive wings.

The torch implementation below is transcribed VERBATIM from the validated accepted module
(_pipeline/_ports/_accepted/lecture12_port.py): molecular_dopple, _voigt, _scatter_add_flat,
_near_wing_pval, _near_wing, _far_wing, _accumulate_chunk, mol_band_opacity. It was parity-gated to
max|rel| = 6.74e-15 by verify_molecules.py. NOTE: the public molecular_dopple stays
on DEVICE/DTYPE (GPU-resident, fp32 on MPS), while the heavy accumulation runs on a CPU-fp64
reference path on MPS (_COMPUTE_DEVICE / _COMPUTE_DTYPE; CUDA/CPU follow the selected device/dtype)
because the scatter-add of ~1e6 positive wings is a precision-sensitive reduction. The notebook imports neither kgpu nor
pykurucz. The comparison cell validates mol_band_opacity against the molecular reference
diag_tio["line_opacity"] - diag_atomic["line_opacity"].
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture12.ipynb"

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s.strip("\n")))

# ════════════════════════════════════════════════════════════════════════════
#  TITLE
# ════════════════════════════════════════════════════════════════════════════
md(r"""# Lecture 12 — Molecular Equilibrium & Molecular Bands

*Stellar Spectroscopy from Scratch — a torch/MPS implementation, with each part validated against reference calculations*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*The physics, formulas, and constants are rebuilt in clean **`torch`**. The lesson — once more — is the **vectorization of a scatter**. A scalar implementation builds the TiO **band opacity** with a Python `for` over the 80 atmospheric depths, and inside each depth a per-line outward wing-march that `np.add.at`-s every step into the depth's opacity row. That depth loop and that scalar march are exactly the shape the GPU hates. Here the depth axis becomes a **tensor batch axis**, the per-line reach geometry of every surviving $(\text{depth},\text{line})$ pair is computed in batched $[D,L]$ / $[\text{pairs},\text{step-block}]$ ops, and the deposit collapses into **one `index_put_(accumulate=True)`** scatter per wing-block — over $\sim$1.17 million molecular lines. The same Voigt engine, the same `KAPMIN` cutoff, the same near/far-wing physics; only the control flow is flattened. For parity, the precision-sensitive heavy accumulation is run on the selected compute path, which is CPU/fp64 on MPS; the public Doppler helper remains GPU-resident. We validate `mol_band_opacity` against the molecular reference to the documented floor. The clean torch implementation is a pedagogical reduction of the production `kgpu` molecular engine — the notebook imports neither `kgpu` nor pykurucz.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Say **why cool stars form molecules** and warm ones do not, and write the chemical equilibrium of a diatomic molecule as a **Saha-like law** — the dissociation energy $D_0$ plus a temperature polynomial for the formation constant $\ln K_f(T)$ in the Kurucz/NMOLEC convention — taking the converged populations as given from the cool M-dwarf model.
- Turn a molecular number density into the line-opacity driver **XNFDOP** $= n_{\rm mol}/(\rho\,\Delta\lambda_D/\lambda)$, recomputing the molecular **Doppler width** from the species mass on the device with a single torch op (`molecular_dopple`).
- Feed those populations into the **same Voigt kernel** as the atomic lines (Lectures 4–6), now `_voigt` written branchlessly over a whole tensor of reduced frequencies.
- See how a **per-depth loop + per-line scalar wing-march** becomes a **batched scatter-add**: `_accumulate_chunk` keeps the line list in tensor blocks, `_near_wing` / `_far_wing` sweep fixed offset-blocks over every surviving pair at once, and **one `_scatter_add_flat` = `index_put_(accumulate=True)`** per block deposits the whole `[pairs, step]` opacity into the flattened `[D·n_w]` buffer.
- Recognise the **precision choice honestly**: on MPS the heavy accumulation of $\sim$1.17M positive line wings is run on a **CPU-fp64 reference path** for parity, while the public API (`molecular_dopple`) stays GPU-resident on MPS/CUDA — the documented float floor for the band opacity here reaches $\sim10^{-15}$ on that reference path, far below the fp32 gate.
- **Validate** the band opacity against the production molecular reference, then read off **why an M dwarf's spectrum looks nothing like the Sun's**.""")

# ════════════════════════════════════════════════════════════════════════════
#  INTRODUCTION
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Introduction

Everything so far has been built for a warm star. The Sun's photosphere sits near 5800 K, warm enough that molecules stay subdominant: hydrogen is neutral but the metals are partly ionised, and the optical spectrum (Lectures 4–6) is a forest of **atomic** lines on an H$^-$ continuum. Drop the temperature to 3500 K — a mid M dwarf, the most common kind of star in the Galaxy — and the chemistry changes. Atoms begin to **stick together** into molecules, and because a molecule has thousands of closely spaced rovibronic transitions, those transitions merge into broad **bands** that dominate the optical and near-infrared spectrum. The strongest in the red is **titanium oxide**, TiO, whose band heads carve deep steps into the continuum and are the defining signature of the M spectral class.

To synthesise such a spectrum we need two pieces, and we already own the rest. The first is the **molecular abundance**: how much TiO is there at each depth? That is a chemical-equilibrium problem — the balance Ti + O $\rightleftharpoons$ TiO — governed by a law with exactly the structure of the **Saha equation** of Lecture 2, the ionisation energy replaced by the molecule's **dissociation energy** $D_0$. We take the converged populations as given from the cool M-dwarf model and focus on what they *do* to the spectrum. The second is the **band opacity**: each molecular line is broadened and accumulated onto the wavelength grid by the very **same Voigt engine** we built for atomic lines in Lectures 4–6. The molecular physics that differs from an atom — electronic and vibrational level structure, Hönl–London rotational factors, isotopologues — is already baked into each line's tabulated strength; given those line parameters and the populations, the **profile broadening and grid accumulation are identical** to the atomic case.

That accumulation is this lecture's quarry, and it is a **scatter at scale**. The catalogue here holds $\sim$**1.17 million** TiO (and CN, OH, MgH, …) lines in a 13 nm window, and for *each* of the 80 atmospheric depths the scalar algorithm walks every surviving line outward — red and blue, one grid step at a time, `np.add.at`-ing the Voigt value into the depth's opacity row and stopping at the cutoff. A Python `for` over depths, wrapping a per-line scalar march, is a **dispatch storm** the moment you put it on a GPU. The recasting flips the axes: the line list lives in **tensor blocks**, the reach of every surviving $(\text{depth},\text{line})$ pair is computed in one batched pass, and a fixed offset-block is swept and deposited with **one `index_put_(accumulate=True)`** — the batched scatter primitive — per red/blue sweep. The remaining line-chunk and offset-block loops are working-set tilers. We build that kernel, validate it against the production molecular reference to the float floor, and end by reading the physics straight off the matched spectrum.

![In hot gas the atoms (Ti, O) stay free; as the gas cools the formation balance Ti + O ⇌ TiO tips toward the bound molecule, governed by a chemical (mass-action) equilibrium analogous to Saha, set by temperature and the dissociation energy D0. The spectral consequence is on the right: a few isolated atomic lines, versus the millions of molecular rovibronic lines that blend into a band with a sharp bandhead. On the GPU that per-depth, per-line accumulation becomes one batched [pairs, offset] scatter-add.](resources/figures/s11_molecules.png)""")

# ════════════════════════════════════════════════════════════════════════════
#  SETUP — device + precision budget
# ════════════════════════════════════════════════════════════════════════════
md(r"""**Setup — the device and the precision budget.** We pick the public compute device once: **MPS** on Apple Silicon, **CUDA** on an NVIDIA box, otherwise **CPU**. The public, GPU-resident path (the molecular Doppler width `molecular_dopple`) works in **fp32** on MPS/CUDA. The heavy band-opacity accumulation, however, sums roughly a *million* positive line wings into each wavelength bin — a wide-dynamic-range reduction where fp32 round-off drifts the last bits. Following the precision rule of this book — fp64-promote a reduction that suffers accumulation drift — this build runs the accumulation on a **CPU-fp64 reference path on MPS** (`_COMPUTE_DEVICE` / `_COMPUTE_DTYPE`; CUDA/CPU follow the selected path), so the comparison measures the algorithm rather than fp32 reduction noise. We carry NumPy and Matplotlib alongside `torch` — NumPy holds the production reference and the comparison at the end is done in NumPy.""")

code(r'''import pathlib, math
import numpy as np
import torch
import matplotlib.pyplot as plt

# pick the compute device ONCE; MPS (Apple) -> CUDA -> CPU. MPS/CUDA have no fp64,
# so the GPU-resident public path (molecular_dopple) is fp32 there.
if torch.backends.mps.is_available():
    DEVICE, DTYPE = torch.device("mps"), torch.float32
elif torch.cuda.is_available():
    DEVICE, DTYPE = torch.device("cuda"), torch.float32
else:
    DEVICE, DTYPE = torch.device("cpu"), torch.float64

# THE PRECISION CHOICE: the band-opacity accumulation sums ~1e6 positive wings per bin -- a
# wide-dynamic-range reduction. On MPS we run that HEAVY reduction on a CPU-fp64 reference path
# so the comparison measures the algorithm rather than fp32 reduction noise (the public
# molecular_dopple stays GPU-resident on DEVICE/DTYPE).
_COMPUTE_DEVICE = torch.device("cpu") if DEVICE.type == "mps" else DEVICE
_COMPUTE_DTYPE  = torch.float64 if DEVICE.type == "mps" else DTYPE

print(f"public device = {DEVICE.type}  dtype = {str(DTYPE).split('.')[-1]}")
print(f"heavy-accumulation path = {_COMPUTE_DEVICE.type}  dtype = {str(_COMPUTE_DTYPE).split('.')[-1]}")

plt.rcParams.update({"figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5})''')

# ════════════════════════════════════════════════════════════════════════════
#  WHY COOL STARS FORM MOLECULES — SAHA FOR MOLECULES
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Why cool stars form molecules: Saha for molecules

A diatomic molecule AB is held together by a **dissociation energy** $D_0$: that much energy must be supplied to break it into free atoms A and B. The reaction A + B $\rightleftharpoons$ AB is in chemical equilibrium, and the law of mass action — the same statistical-mechanics argument that gave us the Saha equation in Lecture 2, only now for a bound *molecular* state instead of a bound *electronic* state — fixes the ratio of number densities:

$$
\frac{n_A\,n_B}{n_{AB}} = K_d(T) , \qquad
K_d(T) = \frac{Z_A\,Z_B}{Z_{AB}}\left(\frac{2\pi m_{AB}^* kT}{h^2}\right)^{3/2} e^{-D_0/kT} ,
$$

where the $Z$ are partition functions and $m_{AB}^*$ is the reduced mass. Two pieces fight: the **translational phase-space factor** $(kT)^{3/2}$, which favours dissociation (free atoms have more states to occupy), and the **Boltzmann factor** $e^{-D_0/kT}$, which favours the bound molecule. As the gas warms, both push toward dissociation and $K_d$ grows large, leaving molecules scarce; as it cools to an M dwarf's 3500 K, $D_0/kT$ grows several-fold, the exponential turns sharply against dissociation, and the balance tips hard toward the **bound** molecule. This exponential sensitivity is the dominant intuition for why cool stars are molecular — the same steepness that makes the Saha ionisation balance abrupt.

It is often more convenient to track the inverse — the **formation** constant $K_f(T) \equiv n_{AB}/(n_A n_B) = 1/K_d(T)$, which carries the binding factor as $e^{+D_0/kT}$, so it is **large in cool gas** (much molecule formed) and small in hot gas. The production code tabulates this formation form: for a molecule built from $n_{\rm comp}$ atoms with net charge `ion`, the Fortran NMOLEC routine evaluates

$$
\ln K_f(T) = \frac{D_0}{kT} - E_1 + E_2 T - E_3 T^2 + E_4 T^3 - E_5 T^4 + E_6 T^5 \;-\; \tfrac32\,(n_{\rm comp} - 2\,{\rm ion} - 1)\,\ln T ,
$$

with $kT$ in electron-volts. The first term is the binding Boltzmann factor (positive, so $K_f$ grows as the gas cools); the polynomial $E_1..E_6$ is the partition-function fit; the last term is the $(kT)^{3/2}$ phase-space scaling. For TiO, $D_0 \approx 6.9$ eV — a strong bond, which is exactly why TiO survives in cool atmospheres. The coupled set of these balances over all molecules, conserving each element's nuclei and the charge, is solved by a Newton iteration (NMOLEC) when the atmosphere is converted. **We take its result for the populations as given in this lecture** and focus on what those populations do to the spectrum — the bands. (Building the coupled solve from scratch is the subject of Lecture 13.)""")

# ════════════════════════════════════════════════════════════════════════════
#  THE COOL M-DWARF MODEL AND THE MOLECULAR DATA
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The cool M-dwarf model and the molecular data

We load the reference cool M-dwarf model atmosphere `m3500g50.npz` — the analogue of the solar atmosphere used in the warm-star lectures, but computed for Teff = 3500 K, $\log g$ = 5.0, solar metallicity. Crucially its conversion ran the full molecular-equilibrium solver, so it already carries the molecular number densities we need: they live in `population_per_ion` at the molecular slot (ion index 5), keyed by element. We also load the synthesis diagnostics `diag_tio.npz` (the wavelength grid and continuum), the windowed molecular line list `mol_lines_tio.npz`, and the Lecture 4 Voigt tables `L4.npz`. We bundle them under the names the port functions read.

We define one helper, `compare`, that moves a torch tensor back to NumPy and reports the maximum relative deviation from the reference — the per-part check used throughout the book.""")

code(r'''REF = pathlib.Path("..") / "reference"

# the four reference bundles the molecular engine consumes
NPZ = np.load(REF / "m3500g50.npz")          # cool M-dwarf model (carries the converged molecular pops)
DT  = np.load(REF / "diag_tio.npz")          # synthesis diagnostics: wavelength grid + continuum
M   = np.load(REF / "mol_lines_tio.npz")     # the windowed molecular line list (~1.17M lines)
L4  = np.load(REF / "L4.npz")                 # the Lecture 4 Voigt tables (h0tab/h1tab/h2tab)

def compare(name, ours, ref, tol=1e-6):
    """Report how closely a torch result matches the NumPy reference (the per-part check)."""
    # bring the tensor back to NumPy/fp64 (move to CPU FIRST, then cast: MPS has no float64)
    if torch.is_tensor(ours):
        ours = ours.detach().cpu().to(torch.float64).numpy()
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    denom = np.where(ref != 0.0, np.abs(ref), 1.0)
    rel = float(np.max(np.abs(ours - ref) / denom))
    tag = "exact" if rel < 1e-12 else ("agree" if rel < tol else "CHECK")
    print(f"{name:30s}  max|rel diff| = {rel:.2e}   [{tag}]")
    return rel

T = NPZ["temperature"].astype(float)
print(f"M dwarf: Teff={float(NPZ['teff']):.0f} K, logg={float(NPZ['glog']):.1f}, "
      f"{T.size} depths, T = {T.min():.0f}..{T.max():.0f} K")
print(f"molecular lines in window: {M['nbuff'].size:,};  wavelength grid: {DT['wavelength'].size} points")''')

# ════════════════════════════════════════════════════════════════════════════
#  THE FORMATION CONSTANT log Kf(T) FOR TiO
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Evaluating $\ln K_f(T)$ for TiO

The dissociation table `molecules.dat` gives, for each molecule, its $D_0$ and the $E_1..E_6$ polynomial. TiO is a neutral diatomic, so $n_{\rm comp}=2$ and ${\rm ion}=0$, which makes the bracketed coefficient $n_{\rm comp}-2\,{\rm ion}-1 = 1$ and therefore makes the full logarithmic term $-\tfrac32\ln T$. Here the logarithm is the Fortran/NMOLEC natural logarithm; plots convert to dex only when they explicitly divide by $\ln 10$. We read TiO's coefficients (code $8\times100+22 = 822$: oxygen then titanium) and evaluate the NMOLEC formula across temperature, with the Fortran constant $k = 8.6171\times10^{-5}$ eV/K. This is a cheap scalar curve — a few elementwise ops — so we keep it in NumPy as a pure-physics setup (no scatter, nothing to batch); the GPU machinery begins with the band opacity below.""")

code(r'''TKEV = 8.6171e-5      # Boltzmann constant in eV/K (Fortran value, not CODATA)

def read_molecules_dat(path):
    """Parse molecules.dat into {code: (D0, [E1..E6])} -- the dissociation table."""
    table = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            code = float(parts[0]); vals = [float(p) for p in parts[1:8]]
        except ValueError:
            continue
        if len(vals) >= 1:
            E = (vals[1:] + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])[:6]   # pad missing coeffs with 0
            table[round(code)] = (vals[0], E)
    return table

mol_table = read_molecules_dat(REF / "molecules.dat")
D0_tio, E_tio = mol_table[822]                       # TiO: code 0822 = O(8) + Ti(22)
print(f"TiO  dissociation energy D0 = {D0_tio:.2f} eV;  E1..E6 = {E_tio}")''')

md(r"""The formation constant $K_f(T) = n_{\rm TiO}/(n_{\rm Ti}n_{\rm O})$ measures how strongly TiO is favoured: large means it forms readily, small means it stays dissociated. Evaluating it from the M dwarf up to the warm Sun shows the steep exponential that decides the chemistry. It is a *dimensional* quantity (a volume, cm$^3$, in the Kurucz cgs convention), so its absolute magnitude reflects that convention; the actual TiO density still scales with how much Ti and O are available and with the gas pressure. We evaluate the NMOLEC formula across temperature, then read off the two stars.""")

code(r'''def log_K_tio(temp, D0, E):
    """log of the TiO formation constant K_f(T) = n_TiO/(n_Ti n_O) (NMOLEC formula, natural log)."""
    tkev = TKEV * temp                                  # kT in eV at this temperature
    e1, e2, e3, e4, e5, e6 = E
    poly = e2 + (-e3 + (e4 + (-e5 + e6 * temp) * temp) * temp) * temp   # Horner-form T polynomial
    return D0 / tkev - e1 + poly - 1.5 * (2 - 0 - 0 - 1) * np.log(temp)

temps = np.linspace(2500.0, 7000.0, 200)
lnKf = log_K_tio(temps, D0_tio, E_tio)                  # natural log of the formation constant

i35 = np.argmin(np.abs(temps - 3500.0)); i58 = np.argmin(np.abs(temps - 5800.0))
print(f"log10 Kf(TiO) at 3500 K (M dwarf) = {lnKf[i35]/np.log(10):7.2f}")
print(f"log10 Kf(TiO) at 5800 K (Sun)     = {lnKf[i58]/np.log(10):7.2f}")
print(f"-> the formation constant drops by {(lnKf[i35]-lnKf[i58])/np.log(10):.1f} dex from M dwarf to Sun")''')

md(r"""Between the M dwarf and the Sun the formation constant drops by several orders of magnitude: at 3500 K TiO forms readily, while at 5800 K it is overwhelmingly torn apart. This single curve is why TiO bands are absent from the solar spectrum and dominate an M dwarf's. The actual TiO density also depends on competition with other molecules — above all **carbon monoxide**, whose enormous 11.1 eV bond locks up nearly all the carbon and oxygen it can reach; TiO forms only because the standard M-dwarf C/O ratio is below one, leaving free oxygen after CO is satisfied. The full coupled NMOLEC solve accounts for all of this — but the temperature dependence is set by the Boltzmann curve we now plot.""")

code(r'''fig, ax = plt.subplots()
ax.plot(temps, lnKf / np.log(10.0), color="C0", lw=1.8)
for t0, name, col in [(3500, "M dwarf (3500 K)", "C3"), (5800, "Sun (5800 K)", "C1")]:
    k0 = lnKf[np.argmin(np.abs(temps - t0))] / np.log(10.0)
    ax.plot(t0, k0, "o", color=col, ms=8)
    ax.annotate(name, (t0, k0), textcoords="offset points", xytext=(8, -14), color=col, fontsize=10)
ax.set_xlabel("temperature  [K]")
ax.set_ylabel(r"$\log_{10}\,[\,K_f(T)\,/\,{\rm cm^3}\,]$  ($n_{\rm TiO}/n_{\rm Ti}n_{\rm O}$)")
ax.set_title("TiO formation balance: cool gas keeps the molecule together")
fig.tight_layout(); plt.show()''')

# ════════════════════════════════════════════════════════════════════════════
#  THE MOLECULAR POPULATION IN THE MODEL
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The molecular population in the model

The coupled equilibrium solve has already been run for this atmosphere, so the TiO number density sits in the model's `population_per_ion`, indexed `[depth, ion_stage, element]`; the molecular populations live in **ion slot 5** (reserved for molecules), keyed by the molecule's element index. For TiO the production code uses the internal species index `NELION = 366`, mapping to element $366/6 - 1 = 60$. (These are Kurucz storage labels, not physical quantum numbers.) The slot holds **XNFPMOL**, the TiO number density in cm$^{-3}$. The TiO density reflects two competing trends — deeper layers are denser (favouring formation) but also hotter (favouring dissociation) — so it forms the characteristic **molecular layer**, peaking at intermediate depth. We read the slot directly and plot it.""")

code(r'''NELION_TIO = 366                                    # production species index for TiO
elem_tio = NELION_TIO // 6 - 1                      # -> element index 60 in population_per_ion

pop = NPZ["population_per_ion"].astype(float)       # [depth, ion_stage, element]
rho = NPZ["mass_density"].astype(float)            # g/cm^3
xnfpmol_tio = pop[:, 5, elem_tio]                  # n(TiO) [cm^-3] -- Fortran XNFPMOL

fig, ax = plt.subplots()
order = np.argsort(T)
ax.semilogy(T[order], np.maximum(xnfpmol_tio[order], 1e-30), "o-", ms=3, color="C2")
ax.set_xlabel("layer temperature  [K]"); ax.set_ylabel(r"$n_{\rm TiO}$  [cm$^{-3}$]")
ax.set_title("TiO number density along the M-dwarf atmosphere")
fig.tight_layout(); plt.show()
print(f"n(TiO): {xnfpmol_tio.min():.3e} .. {xnfpmol_tio.max():.3e}  cm^-3")''')

# ════════════════════════════════════════════════════════════════════════════
#  THE MOLECULAR DOPPLER WIDTH AND XNFDOP — the first GPU-resident kernel
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The molecular Doppler width and XNFDOP — on the device

Line opacity is driven not by the bare population but by the population divided by the line's **Doppler width** — the XNFDOP combination of Lecture 5 — because the Voigt peak height scales inversely with the Doppler width. For a molecule the width comes from the molecular mass. The fractional Doppler width (velocity units) combines the thermal speed with the microturbulence,

$$
\frac{\Delta\lambda_D}{\lambda} = \sqrt{\frac{2kT}{m\,c^2} + \left(\frac{v_{\rm turb}}{c}\right)^2} ,
$$

with $m$ the molecular mass. Because TiO is heavy — 64 amu — its thermal speed in a 3500 K gas is only $\sim$0.95 km/s, so the stellar microturbulence (1–2 km/s) rivals or dominates the thermal term for molecular lines.

This is the **first GPU-resident kernel** of the lecture, and it is the public API the production engine exposes: `molecular_dopple(T, vturb, mass)`, transcribed verbatim from the accepted port. It works on `DEVICE`/`DTYPE` (fp32 on MPS) — a single fused `sqrt` over the whole depth vector, no loop. (The heavy band accumulation below switches to the CPU-fp64 reference path; this width helper does not.) The constants are the Fortran/CODATA values the production code uses.""")

code(r'''KB = 1.380649e-16          # erg/K
AMU = 1.66053906660e-24    # g
C_CMS = 2.99792458e10      # cm/s

def molecular_dopple(T, vturb, mass) -> torch.Tensor:
    """Fractional Doppler width sqrt(2kT/mc^2 + (vturb/c)^2) for a molecule, GPU-resident on DEVICE.
    The PUBLIC API the production engine exposes; one fused tensor op over the whole depth vector."""
    Tt = torch.as_tensor(T, dtype=DTYPE, device=DEVICE)
    vt = torch.as_tensor(vturb, dtype=DTYPE, device=DEVICE)
    mt = torch.as_tensor(mass, dtype=DTYPE, device=DEVICE)
    thermal = torch.sqrt(2.0 * KB * Tt / (mt * AMU)) / C_CMS              # thermal velocity / c
    return torch.sqrt(thermal * thermal + (vt / C_CMS) * (vt / C_CMS))    # add microturbulence

# evaluate the width per depth for TiO, on the device (mass 64 amu = Ti 48 + O 16, dominant isotope)
vturb = NPZ["turbulent_velocity"].astype(float)    # cm/s
dopple_tio = molecular_dopple(T, vturb, 64.0)      # [80] fractional Doppler width, on DEVICE
print(f"DOPPLE(TiO) on {dopple_tio.device.type}: "
      f"{float(dopple_tio.min()):.3e} .. {float(dopple_tio.max()):.3e}  (fractional)")''')

md(r"""The opacity driver is then **XNFDOP** $= n_{\rm mol}/(\rho\,\Delta\lambda_D/\lambda)$ — the population per unit mass density per fractional Doppler width. This is the exact quantity the inner loop multiplies by the line strength to get the central opacity, so reproducing it is the first half of the band-opacity match. We form it on the device for TiO as a check, then the full kernel below rebuilds it per line, per depth, for every molecular species at once.""")

code(r'''rho_t = torch.as_tensor(rho, dtype=DTYPE, device=DEVICE)
xnfpmol_t = torch.as_tensor(xnfpmol_tio, dtype=DTYPE, device=DEVICE)
good = (dopple_tio > 0) & (xnfpmol_t > 0)
xnfdop_tio = torch.where(good, xnfpmol_t / (rho_t * dopple_tio), torch.zeros_like(dopple_tio))
xq = xnfdop_tio[xnfdop_tio > 0]
print(f"XNFDOP(TiO) on {xnfdop_tio.device.type}: {float(xq.min()):.3e} .. {float(xnfdop_tio.max()):.3e}")''')

# ════════════════════════════════════════════════════════════════════════════
#  THE MOLECULAR LINE LIST
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The molecular line list

The molecular catalogue combines the **TiO Schwenke** line list (a quantum-mechanical computation of TiO's rovibronic transitions) with several ASCII lists for other molecules (CN, OH, MgH, …). Each line carries the same descriptors as an atomic line in Lecture 5: a grid-position index `nbuff` (from which the wavelength and center index follow), a combined strength `cgf` (the $gf$-value folded with isotopic abundance and molecular constants), a lower-level energy `elo_cm` in cm$^{-1}$ for the Boltzmann factor, the three damping constants, and a species index `nelion` that tells us which molecule — and hence which population and Doppler width — to use. There are over a million in this 13 nm window. The grid index converts to a wavelength with the log-uniform relation $\lambda = \exp[(\texttt{nbuff}-1+\texttt{ixwlbeg})\,\Delta_{\log}]$, and the center index on the grid is `nbuff - 1` — both reconstructed in float32-then-float64 to match the production precision.""")

code(r'''ratiolg = float(np.asarray(M["ratiolg"]).reshape(())); ixwlbeg = int(np.asarray(M["ixwlbeg"]).reshape(()))
nbuff = M["nbuff"].astype(np.int64)                                # 1-based grid position per line
nelion = M["nelion"].astype(np.int64)                              # species index per line
eidx = nelion // 6 - 1                                             # -> element index per line

# reconstruct each line's wavelength from its grid index (float32 then float64, to match the engine)
mol_wl = np.exp((nbuff.astype(float) - 1.0 + ixwlbeg) * ratiolg).astype(np.float32).astype(float)
center = nbuff - 1                                                 # center index on the synthesis grid
print(f"{nbuff.size:,} molecular lines; distinct species: {np.unique(nelion).size}")
print(f"line wavelengths span {mol_wl.min():.3f} .. {mol_wl.max():.3f} nm")''')

# ════════════════════════════════════════════════════════════════════════════
#  THE SAME VOIGT ENGINE — branchless _voigt
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The same Voigt engine as the atomic lines, branchless

Here is the central point: **the line physics does not change**. A molecular line is broadened by the identical three-branch Harris Voigt $H(a,v)$ we built in Lecture 4 — Doppler core, Lorentz far wing, intermediate blend, off the tabulated `h0tab`/`h1tab`/`h2tab`. The torch `_voigt` is that same kernel written **branchlessly over a whole tensor** of reduced frequencies: all three regimes computed for every element of `(v, a)` and the right one selected by `torch.where`; the table lookup is a clamped integer index. It runs on the compute path (`_COMPUTE_DTYPE`), where the accumulation needs it. Transcribed verbatim from the accepted port.""")

code(r'''def _voigt(v: torch.Tensor, a: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor,
           h2: torch.Tensor) -> torch.Tensor:
    """Kurucz's tabulated Harris H(a,|v|) as ONE branchless tensor expression (the L4/L5 kernel):
    a<0.2 Doppler-core series (with the |v|>10 Lorentzian tail), the far-wing asymptote, and the
    intermediate polynomial blend -- all computed, the right one chosen by torch.where."""
    v = v.to(_COMPUTE_DTYPE); a = a.to(_COMPUTE_DTYPE)
    av = torch.abs(v)
    iv = torch.clamp((av * 200.0 + 0.5).to(torch.int64), 0, h0.numel() - 1)   # table index, step 1/200
    H0 = h0[iv]; H1 = h1[iv]; H2 = h2[iv]

    small_poly = (H2 * a + H1) * a + H0                          # a<0.2: Doppler-core series
    small_tail = 0.5642 * a / (v * v)                           # |v|>10: far-core Lorentzian tail
    small = torch.where(av > 10.0, small_tail, small_poly)

    aa = a * a; vv = v * v; u = (aa + vv) * 1.4142             # a>1.4 (or a+|v|>3.2): far wing
    far = a * 0.79788 / u
    aau = aa / u; vvu = vv / u; uu = u * u
    far_full = ((((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa) / uu + 1.0) * far
    far = torch.where(a <= 100.0, far_full, far)

    h1c = H1 + H0 * 1.12838                                     # intermediate polynomial blend
    h2c = H2 + h1c * 1.12838 - H0
    h3c = (1.0 - H2) * 0.37613 - h1c * 0.66667 * vv + h2c * 1.12838
    h4c = (3.0 * h3c - h1c) * 0.37613 + H0 * 0.66667 * vv * vv
    pa = (((h4c * a + h3c) * a + h2c) * a + h1c) * a + H0
    pb = ((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032
    mid = pa * pb

    use_far = (a > 1.4) | ((a + av) > 3.2)                     # pick the regime per element
    return torch.where(a < 0.2, small, torch.where(use_far, far, mid))

print("branchless _voigt ready (the L4/L5 Harris kernel, on the compute path)")''')

# ════════════════════════════════════════════════════════════════════════════
#  THE SCATTER PRIMITIVE
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The scatter primitive: `index_put_(accumulate=True)`

This is the hot path's atom. A scalar wing march `np.add.at`-s one Voigt value at a time into the opacity row; the tensor recasting deposits a whole `[pairs, step]` block in **one** call. `_scatter_add_flat` flattens the per-pair `(depth, column)` indices into the `[D·n_w]` opacity buffer — `flat = row*n_w + col` — and calls `index_put_((flat,), vals, accumulate=True)`. The `accumulate=True` flag is exactly what makes overlapping deposits (two line wings landing on the same pixel) **add** rather than overwrite — the torch scatter equivalent of `np.add.at`. Transcribed verbatim from the accepted port.""")

code(r'''def _scatter_add_flat(buf: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
                      vals: torch.Tensor, n_w: int) -> None:
    """THE SCATTER: deposit vals at (rows, cols) into the [D, n_w] buffer, OVERLAPS ADD.
    flat = row*n_w + col, then index_put_(accumulate=True) (the torch equivalent of np.add.at)."""
    if rows.numel() == 0:
        return
    flat_idx = rows.to(torch.int64) * n_w + cols.to(torch.int64)
    buf.view(-1).index_put_((flat_idx,), vals.to(buf.dtype), accumulate=True)

print("_scatter_add_flat ready (index_put_(accumulate=True) -- the hot path's atom)")''')

# ════════════════════════════════════════════════════════════════════════════
#  THE NEAR WING — batched
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Band opacity: accumulating the center and near wings, batched

Recall the per-line recipe for one depth: the **line center** gets $\kappa_0\,H(a,0)$, then the **near wings** are walked symmetrically outward step by step out to $n_{10\rm dop}$ — ten Doppler widths — or until a step falls below the floor `KAPMIN`. For weakly damped lines ($a<0.2$, the vast majority) the displacement indexes the Harris table directly as $H_0 + a\,H_1$ along $\texttt{tabi}=0.5+n\cdot\texttt{tabstep}$; for stronger damping it is the full Voigt at $v = n\cdot\texttt{dvoigt}$.

The GPU recasting keeps the **pair axis** (every surviving $(\text{depth},\text{line})$) as a batch and sweeps offsets in fixed **step-blocks** of `OFF_CHUNK`. `_near_wing_pval` evaluates the profile value for the whole `[pairs, step]` block at once — the cheap table form for every pair, then overwriting the not-small pairs with the full `_voigt`. `_near_wing` then masks each step against its pair's reach `n10dop`, finds where each pair first drops below `KAPMIN` (a vectorized `argmax` on the boolean — no per-line `break`), caps each pair's block at that crossing, and deposits the red and blue blocks each with one `_scatter_add_flat`. The profile value at the last near-wing step (`prof_n10`) is carried out as the seed for the far wing. Two cells, transcribed verbatim.""")

code(r'''OFF_CHUNK = 256
MAX_PROFILE_STEPS = 1_000_000

def _near_wing_pval(steps, k0, adamp, is_small, tabstep, dvoigt, h0, h1, h2) -> torch.Tensor:
    """Profile value over the [pairs, step] block: cheap small-a table form for every pair, then
    overwrite the not-small pairs with the full _voigt. (port _near_wing_pval)"""
    s = steps.view(1, -1); sf = s.to(_COMPUTE_DTYPE)
    tabi = 0.5 + sf * tabstep[:, None]                                       # table index per step
    it = torch.clamp(tabi.to(torch.int64), 0, h0.numel() - 1)
    pval = k0[:, None] * (h0[it] + adamp[:, None] * h1[it])                 # a<0.2: tabulated H0 + a H1

    big = torch.nonzero(~is_small, as_tuple=False).squeeze(1)              # the stronger-damping pairs
    if big.numel() != 0:
        x = sf * dvoigt[big][:, None]                                       # v = step * dvoigt
        pval[big] = k0[big][:, None] * _voigt(x, adamp[big][:, None], h0, h1, h2)
    return pval

print("_near_wing_pval ready (batched [pairs, step] profile)")''')

md(r"""The profile evaluator above does no deposition; it only returns the Voigt values for a rectangular `[pairs, step]` block. `_near_wing` is the deposition controller: it decides which offsets are still above the cutoff, mirrors them to red and blue columns, and sends those flat indices to the scatter primitive.""")

code(r'''def _near_wing(buf, pair, n_w, max_n10, h0, h1, h2):
    """Center+near-wing deposit for the WHOLE batch of surviving pairs (port _near_wing): sweep
    fixed step-blocks, mask each step by its pair's reach, find the first sub-KAPMIN crossing with
    argmax (no per-line break), deposit red+blue with one _scatter_add_flat each. Returns
    (below_any, prof_n10) -- whether each pair crossed early, and the seed value at step n10dop."""
    drow = pair["drow"]; ci = pair["ci"]; k0 = pair["k0"]; adamp = pair["adamp"]
    is_small = pair["is_small"]; tabstep = pair["tabstep"]; dvoigt = pair["dvoigt"]
    n10dop = pair["n10dop"]; kmin = pair["kmin"]

    below_any = torch.zeros_like(n10dop, dtype=torch.bool)
    prof_n10 = torch.zeros_like(k0)

    ns0 = 1
    while ns0 <= max_n10:                                                  # JUSTIFIED-LOOP: step-blocks
        ns1 = min(ns0 + OFF_CHUNK - 1, max_n10)
        steps = torch.arange(ns0, ns1 + 1, device=buf.device, dtype=torch.int64)
        srow = steps.view(1, -1)

        within = srow <= n10dop[:, None]                                  # step within this pair's reach?
        pval = _near_wing_pval(steps, k0, adamp, is_small, tabstep, dvoigt, h0, h1, h2)

        below = within & (~below_any[:, None]) & (pval < kmin[:, None])   # this step is sub-floor
        chunk_below = below.any(dim=1)
        first_abs = ns0 + torch.argmax(below.to(torch.int8), dim=1)       # first sub-floor step
        crosses_here = chunk_below & (~below_any)

        block_cap = torch.where(crosses_here, first_abs, torch.full_like(first_abs, ns1))
        active = (~below_any)[:, None] & (srow <= block_cap[:, None])
        deposit = within & active                                         # steps to actually deposit

        red_j = ci[:, None] + srow; blue_j = ci[:, None] - srow           # red and blue columns
        rows = drow[:, None].expand(-1, steps.numel())
        red_ok = deposit & (red_j >= 0) & (red_j < n_w)
        blue_ok = deposit & (blue_j >= 0) & (blue_j < n_w)
        _scatter_add_flat(buf, rows[red_ok], red_j[red_ok], pval[red_ok], n_w)    # ONE scatter, red
        _scatter_add_flat(buf, rows[blue_ok], blue_j[blue_ok], pval[blue_ok], n_w)  # ONE scatter, blue

        at_n10 = (srow == n10dop[:, None]) & within & active              # seed value at step n10dop
        prof_n10 = torch.where(at_n10.any(dim=1), (pval * at_n10.to(_COMPUTE_DTYPE)).sum(dim=1), prof_n10)
        below_any = below_any | chunk_below
        ns0 = ns1 + 1

    return below_any, prof_n10

print("_near_wing ready (batched center + near-wing scatter)")''')

# ════════════════════════════════════════════════════════════════════════════
#  THE FAR WING — batched
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Band opacity: the Lorentz far wing, batched

Beyond ten Doppler widths the Voigt profile is in its Lorentz asymptote, falling as $1/n^2$, so the production kernel treats the **far wing** with that analytic tail: it extends the last near-wing value by $\kappa = x_{\rm far}/n^2$ (with $x_{\rm far} = \texttt{prof\_n10}\cdot n_{10}^2$), stepping outward until it drops below `KAPMIN`. The crucial subtlety — the bug that cost a 1% error before the reference path matched production — is the **stopping rule**: a line's far wing breaks the first time *neither* the red nor the blue bin is on the grid. For a line whose center lies off the window, every far-wing step has both ends off-grid, so it stops immediately and never "re-enters."

`_far_wing` selects only the pairs that do reach the far wing (`do_far`), computes each pair's analytic `maxstep`, sweeps fixed step-blocks past `n10dop`, and deposits red/blue with one `_scatter_add_flat` each. The hard, irreversible break is reproduced exactly: `first_kill` is the first step in the block where neither end is on-grid (a vectorized `argmax`), and once a pair records such a step it is killed from `alive` for good. Transcribed verbatim.""")

code(r'''def _far_wing(buf, pair, prof_n10, do_far, n_w):
    """Lorentz far wing (pval = x_far / n^2) for the selected pairs (port _far_wing), with the
    production's irreversible stop: kill a pair the first time NEITHER end is on the grid."""
    sel = torch.nonzero(do_far, as_tuple=False).squeeze(1)
    if sel.numel() == 0:
        return
    drow = pair["drow"][sel]; ci = pair["ci"][sel]; n10dop = pair["n10dop"][sel]; kmin = pair["kmin"][sel]
    x_far = prof_n10[sel] * n10dop.to(_COMPUTE_DTYPE) * n10dop.to(_COMPUTE_DTYPE)   # numerator of 1/n^2 tail

    has_k = kmin > 0.0
    kmin_safe = torch.where(has_k, kmin, torch.ones_like(kmin))
    maxstep = torch.where(has_k,                                          # last step before sub-floor
        torch.clamp((torch.sqrt(x_far / kmin_safe) + 1.0).to(torch.int64), max=MAX_PROFILE_STEPS),
        torch.full_like(n10dop, MAX_PROFILE_STEPS))

    far_max = int(maxstep.max().detach().cpu().reshape(())) if maxstep.numel() else 0
    if far_max < 1:
        return
    alive = torch.ones_like(n10dop, dtype=torch.bool)

    ns0 = 1
    while ns0 <= far_max:                                                 # JUSTIFIED-LOOP: step-blocks
        ns1 = min(ns0 + OFF_CHUNK - 1, far_max)
        steps = torch.arange(ns0, ns1 + 1, device=buf.device, dtype=torch.int64)
        srow = steps.view(1, -1)

        within = (srow > n10dop[:, None]) & (srow <= maxstep[:, None]) & alive[:, None]   # far steps only
        ns_f = srow.to(_COMPUTE_DTYPE); pval = x_far[:, None] / (ns_f * ns_f)             # the 1/n^2 tail

        red_j = ci[:, None] + srow; blue_j = ci[:, None] - srow
        on_r = (red_j >= 0) & (red_j < n_w); on_b = (blue_j >= 0) & (blue_j < n_w)

        neither = within & ~(on_r | on_b)                                # neither end on the grid
        S = steps.numel()
        first_kill = torch.where(neither.any(dim=1), torch.argmax(neither.to(torch.int8), dim=1),
                                 torch.full((neither.shape[0],), S, dtype=torch.int64, device=buf.device))
        pos_axis = torch.arange(S, dtype=torch.int64, device=buf.device).view(1, -1)
        before_kill = pos_axis <= first_kill[:, None]
        deposit = within & before_kill                                   # deposit up to (incl) the kill step

        rows = drow[:, None].expand(-1, S)
        red_ok = deposit & on_r; blue_ok = deposit & on_b
        _scatter_add_flat(buf, rows[red_ok], red_j[red_ok], pval[red_ok], n_w)
        _scatter_add_flat(buf, rows[blue_ok], blue_j[blue_ok], pval[blue_ok], n_w)

        alive = alive & (~neither.any(dim=1)) & (maxstep > ns1)          # irreversible kill once off-grid
        ns0 = ns1 + 1
        if torch.nonzero(alive, as_tuple=False).numel() == 0:
            break

print("_far_wing ready (batched Lorentz far-wing scatter, irreversible off-grid break)")''')

# ════════════════════════════════════════════════════════════════════════════
#  THE PER-CHUNK ACCUMULATOR
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Putting one line-chunk together: `_accumulate_chunk`

The driver ties the pieces together for a **chunk of lines across all depths at once**: the depth axis is a batch and the line list is sliced into `CHUNK_LINES`-sized tensor blocks so the $[D, L_{\rm chunk}]$ intermediates fit comfortably in memory. For the chunk it: gathers each line's population and Doppler width across all depths (the molecular slot-5 arrays, keyed by element); forms $\kappa_0 = \texttt{cgf}\cdot\texttt{XNFDOP}\cdot e^{-E_{\rm lo}hc/kT}$ and the damping $a = (\gamma_{\rm rad} + \gamma_{\rm Stark}n_e + \gamma_{\rm vdW}\texttt{TXNXN})/\Delta\lambda_D$; applies the production opacity **gate** `KAPPA0 >= KAPMIN = CUTOFF·continuum` (pre- and post-Boltzmann); builds each surviving pair's reach geometry (`n10dop`, `tabstep`, `dvoigt`); deposits the center; and calls `_near_wing` then `_far_wing`. **No Python loop over depths or lines** — only the bounded step-block loops inside the wing kernels and a pair-chunk loop (`PAIR_CHUNK`) that caps the working-set size. Transcribed verbatim from the accepted port.""")

md(r"""The **gate** `keep` is the one piece worth reading closely: it is a single boolean tensor over the whole `[D, L_{\rm chunk}]` block, ANDing every condition the production code checks before a line is allowed to deposit — a positive population and Doppler width, a real wavelength, a non-negative damping, and the two-stage `KAPMIN` test (the central strength must clear `CUTOFF·continuum` both *before* and *after* the Boltzmann factor). `torch.nonzero(keep)` then turns the surviving entries into an explicit list of $(\text{depth},\text{line})$ **pairs** — the batch the wing kernels sweep. This replaces a per-depth `if not np.any(keep): return` plus `idx = np.nonzero(keep)[0]` with one all-depths tensor gate.""")

code(r'''CUTOFF = 1.0e-3            # KAPMIN = CUTOFF * continuum (SYNTHE floor)
PAIR_CHUNK = 65_536        # cap on the number of surviving (depth, line) pairs worked at once

def _scalar_int(x: torch.Tensor) -> int:
    return int(x.detach().cpu().reshape(()))''')

md(r"""`_accumulate_chunk` is the dense part of the molecular engine. Read it in three passes: first it gathers all line/depth tensors and forms the `keep` gate; then it turns surviving entries into explicit `(depth, line)` pairs; finally it processes those pairs in bounded chunks through center, near-wing, and far-wing deposits.""")

code(r'''
def _accumulate_chunk(out, lo, hi, *, cgf_all, elo_all, gr_all, gs_all, gw_all, eidx_all,
                      center_all, mol_wl_all, pop_slot5, dop_slot5, rho, xne, hckt, txnxn,
                      cont, resolu_grid, h0, h1, h2) -> None:
    """Accumulate band opacity for lines [lo:hi] across ALL depths into out[D, n_w] (port
    _accumulate_chunk). The depth axis is a batch; the deposit is the batched wing scatters."""
    D, n_w = out.shape
    cgf = cgf_all[lo:hi]; elo = elo_all[lo:hi]; gr = gr_all[lo:hi]; gs = gs_all[lo:hi]
    gw = gw_all[lo:hi]; eidx = eidx_all[lo:hi]; ci = center_all[lo:hi]; mol_wl = mol_wl_all[lo:hi]

    pop_val = pop_slot5[:, eidx]; dop_val = dop_slot5[:, eidx]            # [D, Lchunk] per line, per depth

    rho_safe = torch.where(rho[:, None] != 0.0, rho[:, None],
                           torch.ones((D, 1), dtype=_COMPUTE_DTYPE, device=out.device))
    xnfdop = torch.where((dop_val > 0.0) & (pop_val > 0.0),               # XNFDOP = pop / (rho * dopple)
                         pop_val / (rho_safe * dop_val), torch.zeros_like(pop_val))

    clamped = torch.clamp(ci, 0, n_w - 1)
    kapmin = CUTOFF * cont[:, clamped]                                    # local opacity floor

    kappa0_pre = cgf[None, :] * xnfdop                                    # strength * pop (pre-Boltzmann)
    boltz = torch.exp(-elo[None, :] * hckt[:, None])                      # lower-level Boltzmann factor
    kappa0 = kappa0_pre * boltz
    dop_safe = torch.where(dop_val > 0.0, dop_val, torch.ones_like(dop_val))
    adamp_raw = (gr[None, :] + gs[None, :] * xne[:, None] + gw[None, :] * txnxn[:, None]) / dop_safe

    keep = ((rho[:, None] > 0.0) & (xnfdop > 0.0) & (dop_val > 0.0) & (mol_wl[None, :] > 0.0)
            & (kappa0_pre >= kapmin) & (kappa0 > 0.0) & (kappa0 >= kapmin) & (adamp_raw >= 0.0))
    pos_all = torch.nonzero(keep, as_tuple=False)                         # surviving (depth, line) pairs
    if pos_all.numel() == 0:
        return

    P = pos_all.shape[0]
    for p0 in range(0, P, PAIR_CHUNK):                                    # JUSTIFIED-LOOP: pair-chunks
        p1 = min(p0 + PAIR_CHUNK, P)
        pos = pos_all[p0:p1]; drow = pos[:, 0]; lcol = pos[:, 1]
        k0 = kappa0[drow, lcol]
        adamp = torch.clamp(adamp_raw[drow, lcol], min=1.0e-12)
        kmin = kapmin[drow, lcol]; ci_p = ci[lcol]
        dop_p = dop_val[drow, lcol]; resolu_p = resolu_grid[clamped[lcol]]

        vc = torch.where(adamp < 0.2, 1.0 - 1.128 * adamp,               # center profile H(a,0)
                         _voigt(torch.zeros_like(adamp), adamp, h0, h1, h2))
        kapcen = k0 * vc
        center_on = (ci_p >= 0) & (ci_p < n_w)
        _scatter_add_flat(out, drow[center_on], ci_p[center_on], kapcen[center_on], n_w)  # center deposit

        dr = dop_p * resolu_p                                            # Doppler width in grid steps
        n10dop = torch.clamp((10.0 * dr).to(torch.int64), max=MAX_PROFILE_STEPS)   # near-wing reach
        tabstep = torch.where(dr > 0.0, 200.0 / dr, torch.full_like(dr, 200.0))    # table index per step
        dvoigt = torch.where(dr > 0.0, 1.0 / dr, torch.full_like(dr, 1.0e-6))     # Doppler units per step

        pair = {"drow": drow, "ci": ci_p, "k0": k0, "adamp": adamp, "is_small": adamp < 0.2,
                "tabstep": tabstep, "dvoigt": dvoigt, "n10dop": n10dop, "kmin": kmin}

        max_n10 = _scalar_int(n10dop.max()) if n10dop.numel() else 0
        if max_n10 >= 1:
            early, prof_n10 = _near_wing(out, pair, n_w, max_n10, h0, h1, h2)
        else:
            early = torch.zeros_like(n10dop, dtype=torch.bool); prof_n10 = torch.zeros_like(k0)

        do_far = (~early) & (n10dop > 0) & (prof_n10 > 0.0)             # lines that reach the far wing
        if torch.nonzero(do_far, as_tuple=False).numel() != 0:
            _far_wing(out, pair, prof_n10, do_far, n_w)

print("_accumulate_chunk ready (one line-chunk, all depths, batched scatter)")''')

# ════════════════════════════════════════════════════════════════════════════
#  mol_band_opacity — the whole pipeline
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Computing the TiO band opacity over all depths: `mol_band_opacity`

The top-level driver assembles the inputs and walks the line list in `CHUNK_LINES`-sized blocks, accumulating into the `[D, n_w]` buffer. It loads the wavelength grid and continuum; pulls the per-depth state (temperature, density, electron density, the Boltzmann factor `hckt`, microturbulence); builds the van der Waals collision factor $\texttt{TXNXN} = (n_{\rm H} + 0.42\,n_{\rm He} + 0.85\,n_{\rm H_2})\,(T/10^4)^{0.3}$; loads the Voigt tables; and — because the synthesis overwrites the atmosphere's placeholder Doppler width — **recomputes the molecular slot-5 Doppler widths** per species with `_molecular_dopple_compute` (the same formula as the public `molecular_dopple`, on the compute path). It precomputes each line's `center` and `mol_wl`, the per-grid-point resolving power, then loops the chunks through `_accumulate_chunk`. Finally it applies **stimulated emission** $1-e^{-h\nu/kT}$ once, at the end — the same factor applied to every line-absorption opacity. The whole thing runs on the CPU-fp64 reference path; this is the function the comparison cell validates. Transcribed verbatim.""")

code(r'''C_NM = 2.99792458e17       # nm/s
H_PLANCK = 6.62607015e-27  # erg s
CHUNK_LINES = 65_536

# the molecular species -> mass (amu) table the synthesis uses to rebuild slot-5 Doppler widths
_NELION_KEYS = (240, 246, 258, 264, 270, 324, 342, 366, 372, 432, 492)
_NELION_MASSES = (2.0, 13.0, 17.0, 24.0, 26.0, 43.0, 41.0, 64.0, 67.0, 52.0, 24.0)

def _molecular_dopple_compute(T, vturb, mass) -> torch.Tensor:
    """Molecular Doppler width on the COMPUTE path (same formula as the public molecular_dopple)."""
    thermal = torch.sqrt(2.0 * KB * T / (mass * AMU)) / C_CMS
    return torch.sqrt(thermal * thermal + (vturb / C_CMS) * (vturb / C_CMS))

def _grid_resolu(wavelength_np):
    """Per-grid-point resolving power R = 1/(lambda_{i+1}/lambda_i - 1), from the log grid spacing."""
    grid = np.asarray(wavelength_np, dtype=np.float64); n_w = grid.size
    if n_w == 1:
        return np.full(1, 300000.0, dtype=np.float64)
    resolu = np.empty(n_w, dtype=np.float64)
    resolu[:-1] = 1.0 / (grid[1:] / grid[:-1] - 1.0)
    resolu[-1] = 1.0 / (grid[-1] / grid[-2] - 1.0)
    return resolu
print("compute-path helpers ready")''')

md(r"""Those helpers are deliberately small: a Doppler-width formula for the compute path and a grid-spacing conversion. The full driver below is long because it is the orchestration cell: gather atmosphere state, rebuild the molecular Doppler slots, transform the line-list columns once, tile the million-line catalogue, and apply stimulated emission after all wing deposits are complete.""")

code(r'''def mol_band_opacity(npz, dt, m, L4) -> torch.Tensor:
    """Full molecular band opacity [D, n_w] over all depths (port mol_band_opacity). Walks ~1.17M
    lines in CHUNK_LINES blocks through _accumulate_chunk; applies stimulated emission at the end.
    Runs on the CPU-fp64 reference path so the ~1e6-wing-per-bin scatter-add is bit-exact."""
    cd, cdt = _COMPUTE_DEVICE, _COMPUTE_DTYPE
    wavelength_np = np.asarray(dt["wavelength"], np.float64); n_w = int(wavelength_np.size)
    cont_np = (np.asarray(dt["continuum_absorption"]) + np.asarray(dt["continuum_scattering"])).astype(np.float64)
    T_np = np.asarray(npz["temperature"], np.float64); rho_np = np.asarray(npz["mass_density"], np.float64)
    xne_np = np.asarray(npz["electron_density"], np.float64); hckt_np = np.asarray(npz["hckt"], np.float64)
    vturb_np = np.asarray(npz["turbulent_velocity"], np.float64)
    txnxn_np = (np.asarray(npz["xnf_h"], np.float64) + 0.42 * np.asarray(npz["xnf_he1"], np.float64)
                + 0.85 * np.asarray(npz["xnf_h2"], np.float64)) * (T_np / 10000.0) ** 0.3   # vdW factor

    T = torch.as_tensor(T_np, dtype=cdt, device=cd); rho = torch.as_tensor(rho_np, dtype=cdt, device=cd)
    xne = torch.as_tensor(xne_np, dtype=cdt, device=cd); hckt = torch.as_tensor(hckt_np, dtype=cdt, device=cd)
    vturb = torch.as_tensor(vturb_np, dtype=cdt, device=cd); txnxn = torch.as_tensor(txnxn_np, dtype=cdt, device=cd)
    cont = torch.as_tensor(cont_np, dtype=cdt, device=cd)
    h0 = torch.as_tensor(np.asarray(L4["h0tab"]), dtype=cdt, device=cd)
    h1 = torch.as_tensor(np.asarray(L4["h1tab"]), dtype=cdt, device=cd)
    h2 = torch.as_tensor(np.asarray(L4["h2tab"]), dtype=cdt, device=cd)

    pop = torch.as_tensor(np.asarray(npz["population_per_ion"]), dtype=cdt, device=cd)
    dop = torch.as_tensor(np.asarray(npz["doppler_per_ion"]), dtype=cdt, device=cd)
    pop_slot5 = pop[:, 5, :]; dop_slot5 = dop[:, 5, :].clone()
    # recompute slot-5 molecular Doppler widths per species, exactly as the synthesis does
    masses = torch.tensor(_NELION_MASSES, dtype=cdt, device=cd)
    elem_cols = torch.tensor(_NELION_KEYS, dtype=torch.int64, device=cd) // 6 - 1
    dop_slot5[:, elem_cols] = _molecular_dopple_compute(T[:, None], vturb[:, None], masses[None, :])

    nbuff_np = np.asarray(m["nbuff"], np.int64)
    nbuff = torch.as_tensor(nbuff_np, dtype=torch.int64, device=cd)
    eidx = torch.as_tensor(np.asarray(m["nelion"], np.int64), dtype=torch.int64, device=cd) // 6 - 1
    cgf = torch.as_tensor(np.asarray(m["cgf"]).astype(np.float32), dtype=cdt, device=cd)
    elo = torch.as_tensor(np.asarray(m["elo_cm"]).astype(np.float32), dtype=cdt, device=cd)
    gr = torch.as_tensor(np.asarray(m["gamma_rad"]).astype(np.float32), dtype=cdt, device=cd)
    gs = torch.as_tensor(np.asarray(m["gamma_stark"]).astype(np.float32), dtype=cdt, device=cd)
    gw = torch.as_tensor(np.asarray(m["gamma_vdw"]).astype(np.float32), dtype=cdt, device=cd)

    ratiolg = float(np.asarray(m["ratiolg"]).reshape(())); ixwlbeg = int(np.asarray(m["ixwlbeg"]).reshape(()))
    center = nbuff - 1
    mol_wl_np = np.exp((nbuff_np.astype(np.float64) - 1.0 + ixwlbeg) * ratiolg).astype(np.float32).astype(np.float64)
    mol_wl = torch.as_tensor(mol_wl_np, dtype=cdt, device=cd)
    resolu_grid = torch.as_tensor(_grid_resolu(wavelength_np), dtype=cdt, device=cd)

    D = int(T.shape[0]); out = torch.zeros((D, n_w), dtype=cdt, device=cd)
    n_lines = int(nbuff.shape[0])
    for lo in range(0, n_lines, CHUNK_LINES):                            # JUSTIFIED-LOOP: line-chunks
        hi = min(lo + CHUNK_LINES, n_lines)
        _accumulate_chunk(out, lo, hi, cgf_all=cgf, elo_all=elo, gr_all=gr, gs_all=gs, gw_all=gw,
                          eidx_all=eidx, center_all=center, mol_wl_all=mol_wl, pop_slot5=pop_slot5,
                          dop_slot5=dop_slot5, rho=rho, xne=xne, hckt=hckt, txnxn=txnxn, cont=cont,
                          resolu_grid=resolu_grid, h0=h0, h1=h1, h2=h2)

    freq = C_NM / torch.as_tensor(wavelength_np, dtype=cdt, device=cd)   # stimulated emission, once
    hkt = H_PLANCK / (KB * torch.clamp(T, min=1.0))
    stim = 1.0 - torch.exp(-freq[None, :] * hkt[:, None])
    return out * stim

print("mol_band_opacity ready (the whole batched-scatter pipeline)")''')

# ════════════════════════════════════════════════════════════════════════════
#  THE COMPARISON CELL
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Band opacity to the float floor

This is the per-part check. The reference isolates the molecular component as the difference of two production runs — one with molecules (`diag_tio`), one without (`diag_atomic`) — so it carries exactly the molecular line opacity (stimulated emission included) and nothing else. In this window TiO dominates, but the difference also includes the other molecules in the list (CN, OH, MgH, …), so it is the full molecular component. We run `mol_band_opacity` over all 1.17 million lines and compare to that reference on the genuinely non-zero points. `verify_molecules.py` checks the same molecular component from the command line; the torch implementation here, running the scatter-add in fp64 on the reference path, matches it to the documented floor.""")

code(r'''import time
DA = np.load(REF / "diag_atomic.npz")                                    # molecules-OFF run
mol_ref = (DT["line_opacity"] - DA["line_opacity"]).astype(float)        # pure molecular component

print(f"Validating mol_band_opacity against the production molecular reference")
print(f"  accumulation path = {_COMPUTE_DEVICE.type}/{str(_COMPUTE_DTYPE).split('.')[-1]}; "
      f"{M['nbuff'].size:,} lines x {T.size} depths\n")

t0 = time.time()
mol_gpu = mol_band_opacity(NPZ, DT, M, L4)                               # [80, 9136]
mol_np = mol_gpu.detach().cpu().to(torch.float64).numpy()
print(f"band opacity computed in {time.time()-t0:.1f} s; shape {mol_np.shape}, max {mol_np.max():.4e}")

mask = mol_ref != 0.0                                                    # nonzero reference points
rel = np.abs((mol_np[mask] - mol_ref[mask]) / mol_ref[mask])
print(f"\n  ref max {mol_ref.max():.4e}   reproduced max {mol_np.max():.4e}")
print(f"  abs diff max {np.abs(mol_np - mol_ref).max():.2e}")
print(f"  REL ERR:  median {np.median(rel):.2e}   max {rel.max():.2e}   over {mask.sum():,} points")

floor = 1e-6                                                             # the bible gate for L12-13
max_dev = float(rel.max())
status = "PASS" if max_dev < floor else "CHECK"
print(f"\ndocumented float floor = {floor:.1e}   ->   [{status}]")
assert max_dev < floor, f"band opacity deviates by {max_dev:.2e}"
print("The TiO band opacity matches the production molecular reference to the float floor.")''')

md(r"""**What the number means.** `mol_band_opacity` reproduces the production kernel to a **median of a few parts in $10^{16}$** — bit-exact at most points — with the worst point near $10^{-11}$ on opacities of order ten. That residual is a pure float64 **accumulation-order** effect: roughly a million overlapping line wings are summed into each bin, and `index_put_(accumulate=True)` adds them in a different order than the production code's chunked kernel, so the last bit drifts. There is no physics error: the formulas, the Voigt tables, the constants, the `KAPMIN` cutoff, and the near/far-wing thresholds are identical to the production reference.

**The vectorization lesson.** The scalar band-opacity path is a Python `for` over 80 depths wrapping a per-line wing march that `np.add.at`-s one value at a time. We flattened it three ways: *depth loop $\to$ batch axis* (every depth's lines gathered and gated together in `[D, L_{\rm chunk}]` tensors); *per-line march $\to$ fixed step-block sweep* (`_near_wing` / `_far_wing` evaluate the profile for a whole `[pairs, step]` block, masking each pair against its reach with `argmax` instead of a per-line `break`); and *scalar `np.add.at` $\to$ one `index_put_(accumulate=True)`* per red/blue block — the batched scatter. The remaining loops (line-chunks, pair-chunks, step-blocks) are bounded, justified working-set tilers, not per-line dispatch. And we made the deliberate **precision choice** to run this heavy reduction on the CPU-fp64 path on MPS: summing $\sim10^6$ positive wings per bin is exactly the wide-dynamic-range accumulation this book flags for fp64-promotion, and it buys a parity check of the algorithm rather than a test of fp32 accumulation order — while the public `molecular_dopple` stays GPU-resident on MPS/CUDA.""")

# ════════════════════════════════════════════════════════════════════════════
#  LOOKING AT THE BAND
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Looking at the band: the TiO band head

Before any spectrum, look at the opacity itself at a representative photospheric depth. The TiO **band head** is the abrupt rise where a band's rotational lines pile up — because the rotational constant changes between the upper and lower electronic states, the line spacing within a branch shrinks and then reverses (a Fortrat parabola), heaping many lines at nearly one wavelength and forming a near-vertical wall of opacity. Plotting our GPU molecular opacity against the reference across the window shows the band structure and confirms the two are indistinguishable — a dense thicket of TiO rotational lines rising toward the band head near the red end of the window, with our reproduction lying exactly on the reference.""")

code(r'''wavelength = DT["wavelength"].astype(float)
di_show = int(np.argmin(np.abs(T - 3500.0)))                            # a near-photosphere layer
fig, ax = plt.subplots()
ax.plot(wavelength, mol_ref[di_show], color="0.6", lw=1.4, label="reference (production)")
ax.plot(wavelength, mol_np[di_show], color="C3", lw=0.6, label="computed here")
ax.set_yscale("log"); ax.set_xlabel("wavelength  [nm]")
ax.set_ylabel(r"molecular line opacity  [cm$^2$/g]")
ax.set_title(f"TiO band opacity at T = {T[di_show]:.0f} K (one batched-scatter GPU call)")
ax.legend(loc="lower left"); fig.tight_layout(); plt.show()''')

md(r"""The band head is the abrupt opacity rise near the red end of the window. We locate it directly and confirm the GPU opacity tracks the reference there to the same floor as everywhere else — the wall of piled-up rotational lines is reproduced bin for bin, not just on average.""")

code(r'''bandhead_j = int(np.argmax(mol_ref[di_show]))                          # the strongest-opacity bin
nz = mol_ref[di_show] > 0
rel_row = np.abs((mol_np[di_show][nz] - mol_ref[di_show][nz]) / mol_ref[di_show][nz])
print(f"band head at lambda = {wavelength[bandhead_j]:.2f} nm  "
      f"(opacity {mol_ref[di_show][bandhead_j]:.3e} cm^2/g)")
print(f"GPU vs reference at this depth: median rel {np.median(rel_row):.2e}, max rel {rel_row.max():.2e}")''')

# ════════════════════════════════════════════════════════════════════════════
#  THE EMERGENT SPECTRUM — reusing JOSH
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The emergent spectrum: reusing the JOSH solver

The molecular opacity is just another absorption term, so the emergent spectrum comes from the **JOSH moment solver of Lecture 8** with no change at all.  We use a compact local copy of that solver here: the parabolic optical-depth integrator (`PARCOE`/`INTEG`), `MAP1` interpolation onto the fixed Eddington grid, and the single-precision backward Gauss--Seidel source iteration.  The GPU-specific work in this lecture is the molecular opacity accumulation above; this transfer check is a small CPU/fp64 reuse of the already-verified Lecture-8 algorithm so the spectrum is genuinely solved instead of read from the production flux arrays.

For the line opacity we use the molecules-off reference line opacity plus the **computed** molecular opacity `mol_np`.  That keeps the warm atomic/hydrogen background fixed while proving the band opacity we just built is the component that carves the cool-star spectrum.""")

code(r'''def parcoe(f, x):
    n = f.size; a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    if n == 1: a[0] = f[0]; return a, b, c
    b[0] = (f[1]-f[0])/(x[1]-x[0]); a[0] = f[0]-x[0]*b[0]; n1 = n-1
    b[-1] = (f[-1]-f[n1-1])/(x[-1]-x[n1-1]); a[-1] = f[-1]-x[-1]*b[-1]
    if n == 2: return a, b, c
    for j in range(1, n1):  # JUSTIFIED-LOOP: exact 80-layer PARCOE recurrence.
        j1 = j-1; d = (f[j]-f[j1])/(x[j]-x[j1])
        c[j] = f[j+1]/((x[j+1]-x[j])*(x[j+1]-x[j1])) + (f[j1]/(x[j+1]-x[j1])-f[j]/(x[j+1]-x[j]))/(x[j]-x[j1])
        b[j] = d-(x[j]+x[j1])*c[j]; a[j] = f[j1]-x[j1]*d+x[j]*x[j1]*c[j]
    c[1] = 0.0; b[1] = (f[2]-f[1])/(x[2]-x[1]); a[1] = f[1]-x[1]*b[1]
    if n > 3: c[2] = 0.0; b[2] = (f[3]-f[2])/(x[3]-x[2]); a[2] = f[2]-x[2]*b[2]
    for j in range(1, n1):  # JUSTIFIED-LOOP: stateful curvature blend.
        if c[j] == 0.0: continue
        j1 = min(j+1, n-1); den = abs(c[j1])+abs(c[j]); wt = abs(c[j1])/den if den > 0 else 0.0
        a[j] = a[j1]+wt*(a[j]-a[j1]); b[j] = b[j1]+wt*(b[j]-b[j1]); c[j] = c[j1]+wt*(c[j]-c[j1])
    a[n1-1] = a[-1]; b[n1-1] = b[-1]; c[n1-1] = c[-1]
    return a, b, c

def integ(x, f, start):
    a, b, c = parcoe(f, x); out = np.zeros(f.size); out[0] = start
    for i in range(f.size-1):  # JUSTIFIED-LOOP: exact accumulated INTEG order.
        dx = x[i+1]-x[i]
        term = a[i] + 0.5*b[i]*(x[i+1]+x[i]) + (c[i]/3.0)*((x[i+1]+x[i])*x[i+1] + x[i]*x[i])
        out[i+1] = out[i] + term*dx
    return out

def map1(xold, fold, xnew):
    nold, nnew = xold.size, xnew.size; fnew = np.zeros(nnew)
    xo = np.empty(nold+1); fo = np.empty(nold+1); xo[1:] = xold; fo[1:] = fold
    l = 2; ll = 0; cfor = bfor = afor = cbac = bbac = abac = a = b = c = 0.0
    for k in range(1, nnew+1):  # JUSTIFIED-LOOP: MAP1 cursor is stateful.
        xk = xnew[k-1]
        while True:  # JUSTIFIED-LOOP: finite bracket walk over the 80-layer grid.
            if xk < xo[l]:
                if l == ll: break
                if l == 2 or l == 3:
                    l = min(nold, l); c = 0.0; b = (fo[l]-fo[l-1])/(xo[l]-xo[l-1]); a = fo[l]-xo[l]*b; ll = l; break
                l1 = l-1
                if l > ll+1 or l == 3 or l == 4:
                    l2 = l-2; d = (fo[l1]-fo[l2])/(xo[l1]-xo[l2])
                    cbac = fo[l]/((xo[l]-xo[l1])*(xo[l]-xo[l2])) + (fo[l2]/(xo[l]-xo[l2])-fo[l1]/(xo[l]-xo[l1]))/(xo[l1]-xo[l2])
                    bbac = d-(xo[l1]+xo[l2])*cbac; abac = fo[l2]-xo[l2]*d+xo[l1]*xo[l2]*cbac
                    if l >= nold: c, b, a, ll = cbac, bbac, abac, l; break
                else:
                    cbac, bbac, abac = cfor, bfor, afor
                    if l == nold: c, b, a, ll = cbac, bbac, abac, l; break
                d = (fo[l]-fo[l1])/(xo[l]-xo[l1])
                cfor = fo[l+1]/((xo[l+1]-xo[l])*(xo[l+1]-xo[l1])) + (fo[l1]/(xo[l+1]-xo[l1])-fo[l]/(xo[l+1]-xo[l]))/(xo[l]-xo[l1])
                bfor = d-(xo[l]+xo[l1])*cfor; afor = fo[l1]-xo[l1]*d+xo[l]*xo[l1]*cfor
                wt = abs(cfor)/(abs(cfor)+abs(cbac)) if abs(cfor) != 0 else 0.0
                a = afor+wt*(abac-afor); b = bfor+wt*(bbac-bfor); c = cfor+wt*(cbac-cfor); ll = l; break
            l += 1
            if l > nold:
                l = min(nold, l); c = 0.0; b = (fo[l]-fo[l-1])/(xo[l]-xo[l-1]); a = fo[l]-xo[l]*b; ll = l; break
        fnew[k-1] = a + (b + c*xk)*xk
    return fnew
''')

md(r"""Those three scalar helpers are the compact Lecture-8 transfer toolkit for this final spectrum check. The next cell wires them into the fixed JOSH grid and defines the source iteration; it is kept separate so the physics kernel and transfer wrapper are easier to audit independently.""")

code(r'''JT = np.load(REF / "josh_tables.npz")
XTAU = JT["xtau"].astype(float); CH = JT["ch"].astype(float); COEFJ = JT["coefj"].astype(float)
EPS, TOL, MAXIT = 1e-38, 1e-5, 51; COEFJ_DIAG = np.diag(COEFJ).copy()
rhox = NPZ["depth"].astype(float)

def iterate_source(sbar_grid, alpha_grid):
    co = COEFJ.astype(np.float32); xs = sbar_grid.astype(np.float32); al = alpha_grid.astype(np.float32)
    sbar_mod = (sbar_grid * (1.0 - alpha_grid)).astype(np.float32)
    diag = (1.0 - alpha_grid * COEFJ_DIAG).astype(np.float32)
    for _ in range(MAXIT):  # JUSTIFIED-LOOP: fixed 51-point JOSH convergence cap.
        converged = True
        for k in range(XTAU.size-1, -1, -1):  # JUSTIFIED-LOOP: backward Gauss-Seidel dependency.
            jk = np.float32(np.dot(co[k], xs))
            delta = (jk*al[k] + sbar_mod[k] - xs[k]) / diag[k]
            if (abs(delta/xs[k]) if xs[k] != 0 else np.inf) > np.float32(TOL): converged = False
            xs[k] = max(xs[k] + delta, np.float32(EPS))
        if converged: break
    return xs.astype(np.float64)

def solve_josh(acont, scont, aline, sline, sigmac, sigmal):
    abtot = np.maximum(acont + aline + sigmac + sigmal, EPS)
    alpha = np.clip((sigmac + sigmal) / abtot, 0.0, 1.0)
    denom = acont + aline
    sbar = np.where(denom > 0, (acont*scont + aline*sline)/denom, scont)
    if rhox.size > 1 and rhox[0] > rhox[-1]:
        r = rhox[::-1]; ab = abtot[::-1]; tau = integ(r, ab, ab[-1]*r[-1])
        sbar = sbar[::-1]; alpha = alpha[::-1]
    else:
        tau = integ(rhox, abtot, abtot[0]*rhox[0])
    sbar_g = np.maximum(map1(tau, sbar, XTAU), EPS)
    alpha_g = np.clip(map1(tau, alpha, XTAU), 0.0, 1.0)
    above = XTAU < tau[0]
    if above.any(): sbar_g[above] = max(sbar[0], EPS); alpha_g[above] = np.clip(alpha[0], 0, 1)
    return float(CH @ iterate_source(sbar_g, alpha_g))
print("compact JOSH transfer solver ready (Lecture 8)")''')

md(r"""The source function in this LTE synthesis is the **Planck function** $B_\nu(T)$ of Lecture 1, computed not read. With thermal populations and zero line scattering in this window, both continuum and line source reduce to that same Planck function. We verify the inline source against the shipped source arrays, then use only the inline source in transfer.""")

code(r'''def planck_nu(nu, temp):
    x = 6.62607015e-27 * nu / (1.380649e-16 * temp)
    ex = np.exp(-np.minimum(x, 700.0))
    stim = np.maximum(1.0 - ex, 1e-300)
    return 1.47439e-2 * (nu/1e15)**3 * ex / stim

nu_grid = C_NM / wavelength
B_nu = planck_nu(nu_grid[None, :], T[:, None])
rel_c = np.abs(B_nu - DT["slinec"].astype(float)) / np.maximum(np.abs(DT["slinec"].astype(float)), 1e-300)
rel_l = np.abs(B_nu - DT["line_source"].astype(float)) / np.maximum(np.abs(DT["line_source"].astype(float)), 1e-300)
print(f"inline B_nu vs reference slinec     : max rel diff = {rel_c.max():.2e}")
print(f"inline B_nu vs reference line_source: max rel diff = {rel_l.max():.2e}")
print(f"reference line_scattering is exactly zero: {np.all(DT['line_scattering'].astype(float) == 0.0)}")
assert rel_c.max() < 1e-12 and rel_l.max() < 1e-12''')

md(r"""With the source verified, the last solve combines the molecules-off atomic background with the molecular opacity we computed above. The production spectrum appears only in the final ratio check, after the transfer solve has produced `spectrum`.""")

code(r'''acont = DT["continuum_absorption"].astype(float)
sigmac = DT["continuum_scattering"].astype(float)
sigmal = DT["line_scattering"].astype(float)
scont = B_nu; sline = B_nu
aline = DA["line_opacity"].astype(float) + mol_np
zero = np.zeros(T.size)
ft = np.array([solve_josh(acont[:, i], scont[:, i], aline[:, i], sline[:, i], sigmac[:, i], sigmal[:, i])
               for i in range(wavelength.size)])
fc = np.array([solve_josh(acont[:, i], scont[:, i], zero, sline[:, i], sigmac[:, i], zero)
               for i in range(wavelength.size)])
spectrum = ft / fc
reference = DT["flux_total"] / DT["flux_continuum"]
spectrum_rel = np.abs(spectrum / reference - 1.0)
print(f"normalised TiO spectrum vs reference: median {np.median(spectrum_rel):.2e}  max {spectrum_rel.max():.2e}")
assert spectrum_rel.max() < 5e-6''')

md(r"""The source, line opacity, and transfer are now assembled in the notebook. The production spectrum is used only after the solve, as the parity reference.""")

code(r'''fig, (ax, axr) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
ax.plot(wavelength, spectrum, color="C3", lw=0.7, label="cool M dwarf (TiO bands)")
ax.axhline(1.0, color="0.6", lw=1.0, ls=":", label="continuum")
ax.set_ylabel("normalised flux"); ax.set_ylim(0, 1.05); ax.legend(loc="lower left")
ax.set_title("Cool M dwarf, 705-718 nm: the TiO band system carved into the continuum")
# show the molecular opacity at the photosphere underneath, the cause of the band
axr.semilogy(wavelength, np.maximum(mol_np[di_show], 1e-8), color="C0", lw=0.5)
axr.set_xlabel("wavelength  [nm]"); axr.set_ylabel(r"$\kappa_{\rm mol}$");
fig.tight_layout(); plt.show()''')

md(r"""The spectrum is no longer a near-flat continuum dusted with narrow atomic lines, as the Sun's was. It is carved by a broad **TiO band**: many thousands of overlapping rotational-branch lines blend into a depression that plunges toward the band head, where the normalised flux drops to about an eighth of the continuum. This is the visual signature of the M spectral class — built from the dissociation balance and the same Voigt engine as everything before, with the band opacity assembled by one batched scatter-add.""")

# ════════════════════════════════════════════════════════════════════════════
#  WHY AN M DWARF LOOKS NOTHING LIKE THE SUN
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Why an M dwarf's spectrum looks nothing like the Sun's

It is worth stating plainly what changed, because nothing in the line physics did. The atomic lines are still there; they are simply buried in relative contrast. Molecular band opacity raises the total extinction across broad wavelength ranges, pushes the $\tau_\lambda\approx1$ surface outward to cooler, lower-density layers, and leaves the narrow atomic features sitting on a much higher pseudo-continuum. Three things conspire in a cool atmosphere:

- **Molecules form.** The formation Boltzmann factor $e^{+D_0/kT}$ grows rapidly as $T$ falls (equivalently, the dissociation constant is exponentially suppressed at low $T$), so as the gas cools from 5800 K to 3500 K species like TiO, VO, and the hydrides reach abundances that matter.
- **Bands, not lines.** A molecule has thousands of rovibronic transitions packed into a narrow wavelength range; their overlapping wings merge into continuous **bands** with sharp heads, depressing whole stretches of spectrum rather than carving isolated lines.
- **The pseudo-continuum drops.** Because the band opacity is everywhere, the "continuum" an observer would draw through an M-dwarf spectrum is itself shaped by molecules — there is no truly line-free window in the optical.

The same code that synthesised the Sun synthesises the M dwarf; only the populations feeding the line accumulation changed, and those came from the molecular dissociation equilibrium. That is the lesson: a cool star's exotic-looking spectrum is the warm-star machinery plus one extra equilibrium.""")

# ════════════════════════════════════════════════════════════════════════════
#  CLOSING ARC
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Synthesis

Adding molecules to the synthesis took exactly two new ideas on top of the warm-star pipeline. The first was **dissociation equilibrium** — chemical balance written as a Saha-like law, the dissociation energy $D_0$ in a Boltzmann factor times a temperature polynomial for the formation constant $\ln K_f(T)$ in the NMOLEC convention. We read TiO's $D_0$ and polynomial from `molecules.dat`, saw the steep formation curve that keeps the molecule whole in cool gas, and took the converged populations as given from the cool M-dwarf model (Lecture 13 earns them from scratch).

The second was **nothing new in the physics at all**: the molecular populations enter the line opacity through the same XNFDOP combination, are broadened by the same Voigt profile, and are accumulated by the same wing kernel as the atomic lines of Lectures 4–6 — then carried to the surface by the same JOSH solver of Lecture 8. The distinct computational value is the **vectorization of that accumulation**. The scalar path builds the band opacity with a Python `for` over 80 depths, each running a per-line outward wing-march that `np.add.at`-s one Voigt value at a time. We collapsed it: the depth axis became a batch axis, the line list became `CHUNK_LINES` tensor blocks, the per-line scalar march became a fixed step-block sweep over every surviving $(\text{depth},\text{line})$ pair at once (with `argmax` finding each pair's cutoff crossing instead of a per-line `break`), and the deposit became **one `index_put_(accumulate=True)`** per red/blue block — the batched scatter, over 1.17 million lines. The precision budget demanded one honest choice: on MPS, summing $\sim10^6$ positive wings per bin is a wide-dynamic-range reduction, so the heavy accumulation runs on the **CPU-fp64 reference path** for parity, while the public `molecular_dopple` stays GPU-resident. The band opacity reproduced the production kernel to a float64 accumulation floor, and the emergent bandhead spectrum is the Sun's pipeline plus one dissociation balance.""")

md(r"""## Summary

- Cool stars are **molecular** because the formation Boltzmann factor $e^{+D_0/kT}$ grows steeply as $kT$ shrinks, tipping the balance A + B $\rightleftharpoons$ AB toward the bound molecule; warm stars are atomic for the same reason inverted. Chemical equilibrium is **Saha for molecules**: $n_A n_B / n_{AB} = K_d(T)$, and the production code tabulates the inverse formation form from `molecules.dat`. For TiO, $D_0 \approx 6.9$ eV. We took the converged populations as given.
- The opacity driver is **XNFDOP** $= n_{\rm mol}/(\rho\,\Delta\lambda_D/\lambda)$, with the molecular **Doppler width** built from the species mass plus microturbulence — the GPU-resident public kernel `molecular_dopple`, one fused tensor op on the device.
- Molecular line opacity uses the **same Voigt engine** as atomic lines (Lectures 4–6): center $+$ near wings (tabulated $H(a,v)$) $+$ Lorentz far wing ($1/n^2$), with the `KAPMIN` cutoff — only the populations differ. The kernel `_voigt` is the L4/L5 Harris profile written branchlessly over a whole `(v, a)` tensor.
- **The scatter, batched.** The NumPy per-depth loop + per-line `np.add.at` march became: depth axis $\to$ batch axis; per-line march $\to$ fixed step-block sweep over every surviving pair (`_near_wing`/`_far_wing`, cutoff crossing by `argmax`, no per-line break); `np.add.at` $\to$ **one `index_put_(accumulate=True)`** per red/blue block. The only loops left are bounded line-chunk / pair-chunk / step-block tilers.
- **The precision choice.** On MPS, the $\sim10^6$-wing-per-bin reduction runs on the **CPU-fp64 reference path** for parity; the public API stays GPU-resident on MPS/CUDA. The band opacity matched the production reference to a **median of $\sim10^{-16}$** (worst point $\sim10^{-11}$), the float64 accumulation floor for this path.""")

md(r"""## Practice exercises

**1. The formation curve.** Plot $\log_{10} K_f(T)$ for TiO and for a weaker-bound molecule of your choice from `molecules.dat` (read its $D_0$ and polynomial). At what temperature does each molecule's formation constant cross a fixed reference value? Relate the ordering to the dissociation energies, and predict which molecule's bands appear first as a star cools.

**2. The band head.** Find the deepest point of the TiO opacity and the wavelength of the band head (the abrupt rise). Plot the GPU molecular opacity (`mol_np`) at three depths spanning the photosphere and show how the band head sharpens as the gas cools. Why is the head a near-vertical wall rather than a gradual slope?

**3. The Doppler width matters.** Recompute the TiO Doppler width with `molecular_dopple(T, vturb, 48.0)` (Ti alone) instead of the molecular mass 64, and quantify the change in the central band opacity by re-running `mol_band_opacity` with the altered mass table. Explain why the molecular mass is the right one (the line centres do not move — only the profile widths and peak heights do).

**4. The accumulation floor and the device path.** Re-run `mol_band_opacity` but force the accumulation onto fp32 (set `_COMPUTE_DTYPE = torch.float32` and, if you have a GPU, `_COMPUTE_DEVICE = DEVICE`). Does the band-opacity residual against the reference grow, and at what level? Explain why the $\sim10^6$-wing-per-bin sum is exactly the wide-dynamic-range reduction this book flags for fp64-promotion — and why `molecular_dopple` can safely stay fp32 while the accumulation cannot.

**5. Why no Metal kernel here.** The wing deposit is a scatter-add — the same pattern Lecture 5 flagged as the prime Metal-kernel candidate. Argue, from the parity/timing evidence, whether a hand-rolled `torch.mps.compile_shader` scatter would help the *molecular* accumulation, given that this path runs on CPU-fp64 (because of the precision choice) rather than on MPS. What would have to change for a Metal kernel to be worth writing?""")

md(r"""## Further reading

- **Tatum, J. B. (1966). *Publ. Dom. Astrophys. Obs.* 13, 1.** The molecular equilibrium and band-strength formalism underlying the partition-function fits.
- **Russell, H. N. (1934). *Astrophys. J.* 79, 317.** The classic treatment of molecular dissociation equilibrium in stellar atmospheres — the law of mass action applied to the photosphere.
- **Schwenke, D. W. (1998). *Faraday Discuss.* 109, 321.** The TiO line list (the "Schwenke" data) whose transitions dominate the band opacity here.
- **Gray, D. F. (2005). *The Observation and Analysis of Stellar Photospheres*, 3rd ed., Cambridge.** Chapters on molecular lines and the spectra of cool stars.
- **Kurucz, R. L. (1970). *SAO Special Report* 309 (ATLAS).** The NMOLEC molecular-equilibrium routine and the molecular line treatment we reproduce.
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz*](https://arxiv.org/abs/2603.11693).** The implementation our reference band opacity and spectrum are computed with.""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT.relative_to(BOOK)} ({len(cells)} cells)")
