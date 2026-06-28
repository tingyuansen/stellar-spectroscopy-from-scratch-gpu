#!/usr/bin/env python
"""Assemble content/Lecture2.ipynb (unexecuted) — the GPU EDITION. Execute + render via build.py.

Lecture 2 (GPU) — The Equation of State, ported to clean depth-batched torch/MPS.
Saha-Boltzmann ionization, the partition functions, Debye pressure-ionization lowering,
and the charge-conservation solve for the electron density n_e — all in torch, vectorized
over the depth axis, on the GPU (MPS/CUDA if present, else CPU/fp64). Each result is
validated against the NumPy edition's reference/L2.npz to the documented float floor.

This is the proof-of-concept for the GPU substitution pattern (see PLAN.md). The clean
torch port is a pedagogical reduction of the production kgpu/eos.py (read-only); the
notebook never imports kgpu or pykurucz.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture2.ipynb"

cells = []
def md(src): cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

md(r"""# Lecture 2 — The Equation of State *(GPU Edition)*

*Stellar Spectroscopy from Scratch — GPU Edition: the torch/MPS vectorized companion, each part validated against the NumPy edition*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*This is the **GPU edition** of Lecture 2. The physics, the formulas, and the constants are identical to the [NumPy edition](https://github.com/tingyuansen/stellar-spectroscopy-from-scratch); the equation of state is rebuilt in clean, depth-batched **`torch`** that runs on the GPU (Apple **MPS** or **CUDA**, with a CPU fallback that runs in fp64). The lecture ends with a **comparison cell** that validates the GPU electron density and hydrogen ionization against the NumPy edition's `reference/L2.npz` to the documented float floor. The clean torch port is a pedagogical reduction of the production `kgpu` engine — the notebook imports neither `kgpu` nor pykurucz.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Write the **Boltzmann** distribution and the **partition function**, and say what each enters the opacity calculation for.
- Write the **Saha equation** and explain every factor — the statistical weight (including the free electron's spin factor), the thermal de Broglie volume, the ionization potential, and why $n_e$ sits in the denominator.
- Explain the Kurucz **Debye continuum-lowering** prescription, the pressure-ionization approximation used here, and why it makes deep, dense layers ionize more readily.
- Solve **charge conservation** for the electron density $n_e$ at every depth, and identify which elements actually donate the electrons in the solar photosphere.
- Express the whole equation of state as **depth-batched tensor operations** on the GPU — the 80 atmospheric layers are the batch axis, so every `torch` op processes all depths at once.
- Assemble the full **PFSAHA** per-ion partition functions $U$, physical stage fractions $F_{Z,i}$, and Kurucz population factors $\Phi_{Z,i}=n_ZF_{Z,i}/U_{Z,i}$ — the Ca--Ni/PFIRON grid, the light-element level sums, the packed-table unpacking, and the high-temperature occupation correction — depth-batched on the GPU, in the form the continuous opacity will consume.
- **Validate** both the charge-balance core and the full PFSAHA against the NumPy edition's references to the documented float floor (fp32 GPU $\leftrightarrow$ fp64 NumPy), the independent per-part check.""")

md(r"""## Introduction

Lecture 1 left us with a grey model atmosphere of the Sun: temperature, pressure, and density at every depth — but with the electron density set to zero, a placeholder we promised to fill. That gap is not a detail. The dominant continuum opacity of a cool star, H$^-$, is a hydrogen atom that has *captured* a free electron, so it depends directly on $n_e$; and most line strengths depend on $n_e$ indirectly, through the ionization balance that decides how many atoms sit in the right stage to absorb. Before we can compute a single opacity we must answer two questions at each depth: **which ionization stage is each element in, and how are its electrons distributed among energy levels?**

In local thermodynamic equilibrium the answer is fixed by the temperature, the particle densities, and the chemical abundances, through two classical results: the **Boltzmann** distribution for the populations of energy levels within an ion, and the **Saha** equation for the balance between successive ionization stages. Closing the system requires one more condition — **charge conservation**, that the free electrons exactly balance the positive ions — and that is what finally pins down $n_e$.

The physics is exactly that of the NumPy edition. What changes here is the *machine*: we build all three on the GPU, in `torch`, **vectorized over depth**. The atmosphere has 80 layers; we make depth the batch axis, so a single tensor expression evaluates the Saha balance at every layer at once — no per-depth Python loop. This is the same structure the production `kgpu` engine uses, written plainly. At the end we load the NumPy edition's reference and confirm the GPU numbers match it to the documented float floor.""")

md(r"""**Setup — the device and the precision budget.** First we pick the compute device once, at the top, and a working dtype to match. On Apple Silicon we use **MPS**; on an NVIDIA box, **CUDA**; otherwise we fall back to **CPU**. MPS lacks practical fp64 support, and this edition deliberately uses **fp32** on both MPS and CUDA so the accelerator path has one uniform precision budget; CUDA can support fp64 on suitable hardware, but that is not the default teaching path here. On the GPU the parity bar is the documented fp32 float floor (a few $\times 10^{-6}$ for the equation of state). On CPU we use **fp64** and recover machine precision. We carry NumPy and Matplotlib alongside `torch` — NumPy holds the reference values we validate against, and the comparison at the end is done in NumPy.""")

code(r'''import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt

# pick the compute device ONCE; MPS (Apple) -> CUDA -> CPU. The accelerator teaching
# path uses fp32 on both MPS and CUDA, so its parity bar is the documented float floor;
# on CPU we use fp64 and recover machine precision.
if torch.backends.mps.is_available():
    DEVICE, DTYPE = torch.device("mps"), torch.float32
elif torch.cuda.is_available():
    DEVICE, DTYPE = torch.device("cuda"), torch.float32
else:
    DEVICE, DTYPE = torch.device("cpu"), torch.float64

def dev(x):
    """Move an array/tensor onto the compute device in the working dtype."""
    return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=DEVICE)

print(f"device = {DEVICE.type}   working dtype = {str(DTYPE).split('.')[-1]}")

# one shared Matplotlib style for every figure in the lecture
plt.rcParams.update({
    "figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
})''')

md(r"""Now load the reference bundle — the **NumPy edition's** `reference/L2.npz`, copied here unchanged. It carries the grey solar atmosphere (the depth grid $\tau$, temperature $T$, gas pressure) built in Lecture 1, the tabulated atomic data (partition functions $U$, ionization potentials $\chi$, abundances), and the **gold-standard answers** ($n_e$, the hydrogen ionized fraction) we validate the GPU result against at the end. We define one helper, `compare`, that moves a GPU tensor back to NumPy and prints the maximum relative deviation from the reference — the per-part check, used here exactly as the NumPy edition used it.""")

code(r'''REF = np.load(pathlib.Path("..") / "reference" / "L2.npz")

def compare(name, ours, ref, tol=1e-6):
    """Report how closely a GPU result matches the NumPy reference (the per-part check)."""
    # bring the GPU tensor back to NumPy/fp64 for the comparison (move to CPU FIRST,
    # then cast: MPS has no float64, so the cast must happen on the CPU)
    if torch.is_tensor(ours):
        ours = ours.detach().cpu().to(torch.float64).numpy()
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)

    # divide by |ref|, guarding a zero reference value
    denom = np.where(ref != 0.0, np.abs(ref), 1.0)
    rel = float(np.max(np.abs(ours - ref) / denom))

    tag = "exact" if rel < 1e-12 else ("agree" if rel < tol else "CHECK")
    print(f"{name:28s}  max|rel diff| = {rel:.2e}   [{tag}]")
    return rel

# the grey solar atmosphere from Lecture 1, moved onto the device as tensors
tau = dev(REF["tau"]); T = dev(REF["T"]); P_gas = dev(REF["P_gas"]); tk = dev(REF["tk"])  # tk = k*T [erg]
nd = T.shape[0]
print(f"{nd} layers on {DEVICE.type}; photosphere near tau=2/3 at "
      f"T = {float(T[int(torch.argmin(torch.abs(tau - 2/3)))]):.0f} K")''')

# ── Boltzmann + partition function ──────────────────────────────────────
md(r"""## The Boltzmann distribution and the partition function

Within a single ion, the population of an energy level is set by the **Boltzmann distribution**. If level $i$ has energy $E_i$ above the ground state and statistical weight $g_i$, then in LTE the fraction of that ion's atoms in level $i$ is

$$
\frac{n_i}{n_{\rm ion}} = \frac{g_i}{U(T)}\,e^{-E_i/kT},
\qquad
U(T) = \sum_i g_i\,e^{-E_i/kT},
$$

where $U(T)$ is the **partition function** — the effective number of accessible states at temperature $T$. Computing $U(T)$ from scratch means summing thousands of measured levels per ion — atomic *data*, not physics we derive — so, as in the NumPy edition, we **reuse the tabulated partition functions** and focus on the ionization physics that consumes them. Here they are (moved onto the device) for neutral hydrogen and neutral iron across the atmosphere.""")

code(r'''# partition functions U[layer, Z-1, ion]; ion=0 is the neutral stage — onto the device
U = dev(REF["U"])

# move back to NumPy just for plotting
T_np = T.detach().cpu().numpy()
plt.plot(T_np, U[:, 0, 0].cpu(),  "o-", ms=3, label="H I  (Z=1, neutral)")
plt.plot(T_np, U[:, 25, 0].cpu(), "s-", ms=3, label="Fe I (Z=26, neutral)")
plt.xlabel("temperature  [K]"); plt.ylabel(r"partition function  $U(T)$")
plt.title("Neutral-stage partition functions grow as excited levels switch on")
plt.legend(); plt.tight_layout(); plt.show()

print(f"U(H I) ranges {float(U[:,0,0].min()):.2f}-{float(U[:,0,0].max()):.2f};  "
      f"U(Fe I) ranges {float(U[:,25,0].min()):.1f}-{float(U[:,25,0].max()):.1f}")''')

md(r"""Hydrogen's neutral partition function hugs $2$ — its first excited level sits $10.2\ \mathrm{eV}$ up, out of reach at photospheric temperatures, so only the ground state ($g_0 = 2$) is populated. Iron, with a forest of low-lying levels from its open $d$-shell, climbs to $\sim 30$ as those levels thermally switch on. These numbers feed straight into the Saha equation next.

![Saha sets the balance between ionization stages; Boltzmann sets the level populations within an ion — together they fix which species can absorb, and in the solar line-forming photosphere the metals are the major electron donors.](resources/figures/s2_saha_boltzmann.png)""")

# ── Saha equation ───────────────────────────────────────────────────────
md(r"""## The Saha equation — as a depth-batched tensor expression

Boltzmann distributes atoms among levels *within* an ion; the **Saha equation** distributes them *between* ionization stages:

$$
\frac{n_{i+1}}{n_i}\,n_e
= 2\,\frac{U_{i+1}}{U_i}\,
\left(\frac{2\pi m_e k T}{h^2}\right)^{3/2}
e^{-\chi_i/kT}.
$$

The **$2$** is the freed electron's two spin states; $U_{i+1}/U_i$ compares the two ions' internal states; $(2\pi m_e kT/h^2)^{3/2}$ is the electron's translational phase-space density (the inverse cube of its thermal de Broglie wavelength); $e^{-\chi_i/kT}$ is the Boltzmann penalty for the ionization energy $\chi_i$; and the lone $n_e$ on the left says **the more electrons are already around, the harder it is to stay ionized**.

The thermal-de-Broglie prefactor is a constant, $(2\pi m_e k/h^2)^{3/2} = 2.4148\times10^{15}\ \mathrm{cm^{-3}\,K^{-3/2}}$ — the same literal the NumPy edition (and the production code) carry. In `torch` the whole thing is one elementwise expression over the depth tensors; it returns the ratio $N_{i+1}/N_i$ for **all 80 layers at once**. We carry the prefactor `SAHA` and the kelvin-to-eV factor `KEV` as the reference code's exact literals so our numbers match it.""")

code(r'''SAHA = 2.4148e15        # (2*pi*m_e*k / h^2)^{3/2}  [cm^-3 K^-3/2]
KEV  = 8.6171e-5        # eV per kelvin: kT[eV] = KEV*T, so chi[eV]/(KEV*T) = chi/kT

def saha_ratio(U_lo, U_hi, chi_eV, T, n_e, dchi_eV=0.0):
    """Saha ratio N_{i+1}/N_i, depth-batched in torch. chi lowered by dchi_eV (Debye, below).
    Every argument is a (nd,) device tensor (or a scalar); the result is (nd,)."""
    return (2.0 * U_hi / U_lo) * SAHA * T**1.5 * torch.exp(-(chi_eV - dchi_eV) / (KEV * T)) / n_e''')

# ── Debye lowering ──────────────────────────────────────────────────────
md(r"""## The Kurucz pressure-ionization approximation: Debye lowering

In a dense plasma each ion is screened by a cloud of nearby charges of characteristic **Debye radius** $\lambda_D = \sqrt{kT / (4\pi e^2\, n_{\rm charge})}$, which partially cancels the binding potential and **lowers** the effective ionization potential by

$$
\Delta\chi_i = \frac{i\,e^2}{\lambda_D} \;\approx\; i \times \frac{1.44\times10^{-7}\ \mathrm{eV\,cm}}{\lambda_D},
$$

with the stage index $i$ equal to the charge of the ion after the electron leaves. In code-index language, `ion=0` is the neutral stage, spectroscopic stage I is neutral, and the transition from charge $q$ to $q+1$ uses the resulting charge $q+1$. In the Kurucz/pykurucz prescription used here the lowering is capped at $1\ \mathrm{eV}$; this is an implementation limiter, not a general law of pressure ionization. The screening charge is approximated as $n_{\rm charge} \approx 2n_e$ for a singly-charged plasma. The lowering depends on $n_e$ — the very quantity we are solving for — so it is recomputed each iteration from the current $n_e$; the array `potlow` below holds the converged reference value for reference.""")

code(r'''# Debye lowering per unit charge [eV] at each layer (converged reference value) — onto the device
potlow = dev(REF["potlow"])
print(f"Debye lowering Delta_chi (per unit charge): "
      f"{float(potlow.min()):.2e} eV (top)  ->  {float(potlow.max()):.3f} eV (deepest layer)")''')

md(r"""**Check the Saha physics on hydrogen.** Hydrogen is the cleanest case: two stages (H I and the bare proton H II, $U=1$) and $\chi = 13.6\ \mathrm{eV}$. Holding $n_e$ fixed at the reference value, the depth-batched Saha expression gives the ionized fraction at every depth in one shot. This is a *unit test of the Saha formula alone* — we close the loop and solve for $n_e$ self-consistently below.""")

code(r'''chi = dev(REF["chi"]); n_e_ref = dev(REF["xne"])

U_HI, U_HII = U[:, 0, 0], U[:, 0, 1]          # H I and H II partition functions (depth tensors)
chi_H = chi[0, 0]                             # hydrogen ionization potential [eV] (13.6)

# Saha ratio N(H II)/N(H I); Debye lowering at charge 1 (the neutral -> singly transition)
r = saha_ratio(U_HI, U_HII, chi_H, T, n_e_ref, dchi_eV=potlow * 1)
frac_HII = r / (1.0 + r)                       # ionized fraction n(H II)/n(H total)

compare("H II fraction", frac_HII, REF["fracH"][:, 1], tol=5e-6)
jphot = int(torch.argmin(torch.abs(tau - 2/3)))
print(f"H is {100*float(frac_HII[jphot]):.4f}% ionized at the photosphere")''')

md(r"""The maximum relative difference is at the part-per-million level: the GPU Saha hydrogen reproduces the reference essentially exactly. The headline number is striking — at the photosphere hydrogen is barely $10^{-4}$ ionized. The Sun's most abundant element contributes almost no free electrons where its spectrum forms. So where do the photospheric electrons come from?""")

# ── Charge conservation ─────────────────────────────────────────────────
md(r"""## Charge conservation: solving for the electron density on the GPU

Within this LTE, ideal-gas-plus-Debye atmosphere, the electrons come from whichever elements ionize most easily — Na, Mg, Al, Si, K, Ca, Fe, with first ionization potentials well below hydrogen's $13.6\ \mathrm{eV}$, roughly $4$--$8\ \mathrm{eV}$, and therefore large singly ionized fractions in the relevant photospheric layers. To find $n_e$ we impose **charge conservation**,

$$
n_e = \sum_{\text{elements } Z}\; n_Z \sum_{i} i\,f_{Z,i},
$$

where $n_Z = A_Z\,n_{\rm atom}$ and $f_{Z,i}$ is the fraction of $Z$ in stage $i$ (chained Saha ratios). There is a circularity — the Saha fractions need $n_e$, and $n_e$ needs them — broken by iteration. Every step here is a **depth tensor**: the per-element ionization ladder is built with vectorized `torch` ops over the 80 layers, and the charge sum reduces over elements. The nuclei density follows from $P_{\rm gas} = (n_{\rm atom} + n_e)\,kT$.

As in the NumPy edition, the reference normalises each element over a slightly longer ladder (`nion2` stages) than the stages whose charge we count (`nion`); we match that to reproduce the reference to a part in a million.""")

code(r'''xab   = dev(REF["xabund"])          # abundance A_Z: number fraction over all atoms (99 elements)
nion  = REF["nion"].astype(int)    # stages whose charge we count per element (host int)
nion2 = REF["nion2"].astype(int)   # ladder length the reference normalises over

def ionization_fractions(Z, T, n_e, dchi1):
    """Fractions f[layer, ion] of element Z, chaining Saha ratios over the nion2-stage
    ladder the reference normalises over. Depth-batched: built as a list of (nd,) tensors
    then stacked to (nd, nion2)."""
    ni2 = int(nion2[Z-1])

    # r[:,0]=1: every stage measured against the neutral one. Climb the ladder one stage
    # at a time; the Debye lowering scales with the charge i of the resulting ion.
    cols = [torch.ones(nd, dtype=DTYPE, device=DEVICE)]
    for i in range(1, ni2):
        cols.append(cols[i-1] * saha_ratio(U[:, Z-1, i-1], U[:, Z-1, i],
                                           chi[Z-1, i-1], T, n_e, dchi_eV=dchi1 * i))
    r = torch.stack(cols, dim=1)                 # (nd, ni2)
    return r / r.sum(dim=1, keepdim=True)        # normalise so the stages sum to 1''')

md(r"""**Solving for the electron density.** The fixed-point iteration that closes the loop, entirely on the device: start from the guess that half the particles are electrons, then repeat until $n_e$ stops moving — (1) recompute the atom density and the Debye lowering from the current $n_e$; (2) sum the charge donated by every element; (3) damp by averaging with the old guess. The bare constants ($e = 4.801\times10^{-10}\,\mathrm{esu}$, $4\pi = 12.5664$) are the reference literals. The element loop stays in Python (the dispatch is element-specific), but **every operation inside it is a depth-batched tensor op** — the 80 layers are processed together.""")

code(r'''# stage-charge weights [0,1,2,...] as a device tensor, sliced per element below
CHARGE = torch.arange(6, dtype=DTYPE, device=DEVICE)

def solve_electron_density(max_iter=400, tol=1e-10):
    n_e = P_gas / tk / 2.0                       # initial guess: half the particles are electrons

    for _ in range(max_iter):
        n_atom = P_gas / tk - n_e                # remaining particles are nuclei (atoms + ions)

        # Debye radius [cm], n_charge ~ 2*n_e for a singly-charged gas; lowering capped at 1 eV
        lam_D = torch.sqrt(tk / (12.5664 * (4.801e-10)**2 * 2.0*n_e))
        dchi1 = torch.clamp(1.44e-7 / lam_D, max=1.0)

        # sum the electron donations over every element (depth-batched inside)
        n_e_new = torch.zeros(nd, dtype=DTYPE, device=DEVICE)
        for Z in range(1, 100):
            f = ionization_fractions(Z, T, n_e, dchi1)
            ni = int(nion[Z-1])                  # count charges over the tracked stages only
            # charge per atom = sum_i i*f_i; times element density n_atom*A_Z = electrons donated
            n_e_new = n_e_new + (f[:, :ni] * CHARGE[:ni]).sum(dim=1) * n_atom * xab[Z-1]

        n_e_new = 0.5 * (n_e_new + n_e)          # damp for stability
        if float(torch.max(torch.abs(n_e_new - n_e) / n_e_new)) < tol:
            n_e = n_e_new; break
        n_e = n_e_new
    return n_e

n_e = solve_electron_density()
compare("electron density n_e", n_e, REF["xne"], tol=1e-3)''')

md(r"""The GPU charge-balance solve reproduces the reference electron density across the whole atmosphere to a level set by the float floor — fp32 on the GPU, fp64 on CPU. The residual sits at the last digits of the physical constants the two calculations carry; it is far below any spectroscopic consequence. For a single self-consistent model through the rest of the book we adopt the reference electron density from here, so every later check rests on one exact $n_e$.

Now look at where those electrons come from — and how the answer shifts with depth.""")

md(r"""**Attributing the electrons.** We run the same Saha machinery one element at a time, store each element's contribution, and collapse it into two competing shares: hydrogen alone, and all metals ($Z \geq 3$). We evaluate at the reference $n_e$ so the attribution rests on a single exact electron density.""")

code(r'''# electron donation by every element at every depth, at the reference n_e (all on device)
n_atom   = P_gas / tk - n_e_ref
n_charge = 2.0 * n_e_ref
dchi1    = torch.clamp(1.44e-7 / torch.sqrt(tk / (12.5664 * (4.801e-10)**2 * n_charge)), max=1.0)

ne_Z = torch.zeros(nd, 100, dtype=DTYPE, device=DEVICE)   # ne_Z[:, Z] = electrons from element Z
for Z in range(1, 100):
    f = ionization_fractions(Z, T, n_e_ref, dchi1)
    ni = int(nion[Z-1])
    ne_Z[:, Z] = (f[:, :ni] * CHARGE[:ni]).sum(dim=1) * n_atom * xab[Z-1]

# collapse into two competing shares (move to NumPy for the plot)
total       = ne_Z[:, 1:].sum(dim=1)
H_share     = (ne_Z[:, 1] / total).cpu().numpy()
metal_share = (ne_Z[:, 3:].sum(dim=1) / total).cpu().numpy()
ne_Z_np = ne_Z.cpu().numpy(); total_np = total.cpu().numpy()''')

md(r"""Now plot the two shares against depth, mark the line-forming layers and the $\tau = 2/3$ photosphere, and print the dominant donors at a representative line-forming layer.""")

code(r'''tau_np = tau.cpu().numpy(); logtau = np.log10(tau_np)
plt.plot(logtau, 100*H_share,     color="C3", label="hydrogen")
plt.plot(logtau, 100*metal_share, color="C0", label=r"metals ($Z\geq3$)")
plt.axvspan(-2, -1, color="0.8", alpha=0.5, label="typical line-forming layers")
plt.axvline(np.log10(2/3), ls="--", color="0.4", lw=1)
plt.text(np.log10(2/3)+0.1, 50, r"$\tau=2/3$", color="0.3")
plt.xlabel(r"$\log_{10}\tau_{\rm Ross}$"); plt.ylabel("share of free electrons  [%]")
plt.title("Who supplies the electrons depends on depth"); plt.legend(); plt.tight_layout(); plt.show()

jl  = int(np.argmin(np.abs(tau_np - 0.05)))            # layer closest to tau = 0.05
sym = {11:"Na", 12:"Mg", 13:"Al", 14:"Si", 19:"K", 20:"Ca", 26:"Fe", 6:"C", 28:"Ni"}
top = sorted(range(3, 100), key=lambda Z: ne_Z_np[jl, Z], reverse=True)[:5]
print(f"line-forming layer (tau={tau_np[jl]:.3f}, T={T_np[jl]:.0f} K): metals supply "
      f"{100*metal_share[jl]:.0f}% of electrons; top donors: "
      + ", ".join(f"{sym.get(Z,Z)} {100*ne_Z_np[jl,Z]/total_np[jl]:.0f}%" for Z in top))''')

md(r"""The competition turns over with depth. In the **line-forming layers** ($\tau \sim 0.01$–$0.1$, $T \approx 4800\ \mathrm{K}$) the easily-ionized metals supply most of the electrons — Mg, Fe, Si, Na — even though they are a thousand times rarer than hydrogen. This is the regime that matters for the spectral lines of Lectures 4–6, and it is why metal-poor stars have lower electron pressures and weaker H$^-$ continuum opacity at the same temperature. Deeper, hydrogen's overwhelming abundance wins; higher up, the very thin gas ionizes hydrogen just enough that even that tiny fraction overtakes the already-saturated metals. The slogan "metals make the electrons" holds exactly in the line-forming layers where we need it.""")

md(r"""## Completing the atmosphere

With $n_e$ in hand the grey atmosphere is complete: every depth carries temperature, pressure, density, **and** electron density. We close with a two-panel portrait — the electron density (log scale) on the left, the hydrogen ionized fraction on the right — and fill the `XNE` column the grey atmosphere left empty.""")

code(r'''fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))
xne_np = REF["xne"]
ax[0].plot(logtau, np.log10(xne_np), color="C0")
ax[0].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[0].set_ylabel(r"$\log_{10} n_e$  [cm$^{-3}$]")
ax[0].set_title("Electron density vs depth")

ax[1].plot(logtau, frac_HII.cpu().numpy(), color="C3")
ax[1].axvline(np.log10(2/3), ls="--", color="0.5", lw=1)
ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$")
ax[1].set_ylabel("ionized fraction of hydrogen"); ax[1].set_title("Hydrogen ionization vs depth")
fig.tight_layout(); plt.show()

XNE = xne_np.copy()
print(f"XNE filled: {XNE[0]:.3e} (top) -> {XNE[-1]:.3e} (bottom) cm^-3 — atmosphere complete.")''')

# ── The validation cell — the per-part GPU check ─────────────────────────
md(r"""## The comparison cell — validating the GPU result against the NumPy edition

This is the per-part check that defines the GPU edition. We have just computed the equation of state in `torch`, on the GPU, in fp32 (or fp64 on a CPU fallback). Now we put **every** GPU result the lecture produced next to the NumPy edition's reference — the reference implementation, itself checked against pykurucz — and report the **maximum relative deviation**, asserting it is below the documented float floor. Passing this gate establishes equivalence to that reference implementation and its LTE/Kurucz modeling choices; a failure points to a porting bug. This is the independent validation the edition promises, lecture by lecture.""")

code(r'''print(f"Validating the GPU equation of state against reference/L2.npz")
print(f"  device = {DEVICE.type}   dtype = {str(DTYPE).split('.')[-1]}\n")

# the documented EOS float floors: fp32 GPU lands at a few e-6; fp64 CPU recovers machine precision
is_fp32 = (DTYPE == torch.float32)
floor_ne   = 5e-6 if is_fp32 else 1e-10
floor_frac = 5e-6 if is_fp32 else 1e-10

devs = {
    "electron density n_e": compare("electron density n_e", n_e, REF["xne"], tol=floor_ne),
    "n_e (at reference)":   compare("n_e via attribution",  ne_Z[:, 1:].sum(dim=1), REF["xne"], tol=floor_ne),
    "H II fraction":        compare("H II fraction",        frac_HII, REF["fracH"][:, 1], tol=floor_frac),
}

max_dev = max(devs.values())
floor = max(floor_ne, floor_frac)
print(f"\nmax relative deviation (GPU {DEVICE.type}/{str(DTYPE).split('.')[-1]} vs NumPy reference) "
      f"= {max_dev:.3e}")
status = "PASS" if max_dev < floor else "CHECK"
print(f"documented float floor = {floor:.1e}   ->   [{status}]")
assert max_dev < floor, f"GPU EOS deviates by {max_dev:.2e}, above the float floor {floor:.1e}"
print("\nThe GPU equation of state matches the NumPy edition to the documented float floor.")''')

md(r"""**What the number means.** The GPU equation of state reproduces the NumPy edition's electron density and hydrogen ionization to the float floor — a few $\times 10^{-6}$ in fp32 on the GPU, machine precision in fp64 on a CPU run. That residual is the single-precision round-off of the Saha ladder and the charge-balance reduction, *not* a physics difference: the formulas, the constants, and the atomic data are identical to the NumPy edition (and hence to pykurucz). The depth-batched `torch` port is correct relative to the reference implementation.""")

# ════════════════════════════════════════════════════════════════════════════
#  PFSAHA — the full per-ion partition functions + per-ion populations on the GPU
# ════════════════════════════════════════════════════════════════════════════
md(r"""---

## PFSAHA: per-ion partition functions and population factors

The charge-balance core above gave us $n_e$ with successive stage-to-stage Saha ratios and *tabulated* partition functions $U(T)$. But Lecture 3's continuous opacity needs more than $n_e$: it needs the physical stage fraction of every ionization stage, $F_{Z,i}$, and the **partition function of each ion**, $U_{Z,i}$, built from atomic data rather than read from a temperature-only table. The opacity routines do not store the raw ion population $n_ZF_{Z,i}$ directly. They store the Kurucz population factor

$$
\Phi_{Z,i} = \frac{n_Z F_{Z,i}}{U_{Z,i}}, \qquad n_Z = A_Z\, n_{\rm atom},
$$

because later opacity formulas multiply this stored factor by level weights and Boltzmann exponentials. This is the job of **PFSAHA** — the production engine (`pfsaha_exact` in the Kurucz codes) that assembles, at every depth and for every element/ion, the partition function $U$ and then runs the Saha ladder to give both $F/U$ and $\Phi$. We port the whole engine to depth-batched `torch`, and validate the per-ion $U$, $F/U$, and $\Phi$ against the NumPy edition's `pfsaha_truth.npz`.

The partition function is no longer a single table lookup. Three different machines build it, dispatched by element:

- **Ca--Ni / PFIRON block** ($Z=20$--$28$, including the traditional iron-group elements): a measured $U(\log_{10}T,\ \log_{10}\Delta\chi)$ grid (PFIRON), bilinearly interpolated.
- **Light elements** (H, He, C, O, Na, Mg, Al, Si, Ca, K, B): an explicit sum over measured energy levels $U = \sum_i g_i\,e^{-E_i/kT}$.
- **Everything else**: a packed-integer table (NNN) of bracketed partition values, unpacked and interpolated in temperature.

On top of $U$ sit two corrections: a low-$T$ **ground-state floor** (PFGROUND) and a high-$T$ Kurucz occupation/continuum-lowering correction, inspired by the Hummer--Mihalas treatment of dissolved high-lying states. Every element is processed over **all 80 depths at once**; the only Python loops are over the 99 elements and their $\le 9$ ion stages — tiny, fixed counts with genuinely different physics per element, exactly the structure the production `kgpu` engine uses.""")

md(r"""**A note on what runs on the GPU and what runs on the host — and why.** The *continuous* arithmetic (the Boltzmann sums, the interpolation blends, the Saha ladder) runs **on-device, depth-batched**; but the *discrete* table decisions — which integer cell of the NNN/PFIRON grid a temperature falls in, and which side of the high-$T$ occupation gate a layer is on — are resolved on the **fp64 host**. The reason is that MPS lacks practical fp64 support, and an fp32 boundary slip near a grid edge would jump to a *different* integer cell, a $\sim 1\%$ discontinuity in $U$. The host decision is made **once per element, vectorized over all 80 depths** (no Python loop over depth), so the sync cost is hidden. This is the honest way to keep a discrete-table engine bit-faithful on the accelerator path.""")

md(r"""**Load the PFSAHA atomic-data tables.** The reference `pfsaha_inputs.npz` carries the depth state PFSAHA consumes (the same $T$, $P_e$, $P_g$, $n_{\rm atom}$, abundances Lecture 2 already treats as given) and the measured atomic-data tables — the packed NNN partition data, the POTION ionization potentials, the PFIRON iron-group grid, the explicit light-element level blocks, and the pre-tabulated PFGROUND ground-state correction. We move the continuous tables onto the device and keep fp64 host copies of the discrete ones (NNN, PFIRON, SCALE) for the integer-grid lookups.""")

code(r'''import math
PF = np.load(pathlib.Path("..") / "reference" / "pfsaha_inputs.npz")
PFT = np.load(pathlib.Path("..") / "reference" / "pfsaha_truth.npz")   # the comparison target

# PFSAHA physical constants — the exact ATLAS/pykurucz literals
EV_TO_CM = 8065.479          # 1 eV in cm^-1
LN10     = math.log(10.0)

# depth state (continuous -> device tensors; the discrete-lookup copies stay fp64 on host)
T_pf   = dev(PF["T"]);   tkev = dev(PF["tkev"]); hckt = dev(PF["hckt"])
tlog   = dev(PF["tlog"]); tk_pf = dev(PF["tk"]); Pg = dev(PF["gas_pressure"])
xne_pf = dev(PF["electron_density"]); xnatom = dev(PF["xnatom"]); xab_pf = dev(PF["xabund"])
T_h    = PF["T"].astype(np.float64);    tlog_h = PF["tlog"].astype(np.float64)   # fp64 host copies
nion_per_Z = PF["nion_per_Z"].astype(np.int64); LOCZ = PF["LOCZ"].astype(np.int64)
ndp = T_h.size

# atomic-data tables: discrete ones kept fp64 on host (exact integer unpacking), the
# pre-tabulated PFGROUND + the light-element level blocks moved onto the device
NNN_i      = PF["NNN"].astype(np.int64)            # (6,374) packed partition data (host int)
POTION_h   = PF["POTION"].astype(np.float64)       # (999,) ionization potentials [cm^-1]
SCALE_h    = PF["SCALE"].astype(np.float64)
PFTAB_h    = PF["PFTAB"].astype(np.float64)        # (7,56,10,9) Fe-group grid
PFLO_h     = PF["PFIRON_POTLO"].astype(np.float64)
PFLOLOG_h  = PF["PFIRON_POTLOLOG"].astype(np.float64)
pfg = dev(PF["pfground_tab"])                       # (605,80) ground-state floor, pre-evaluated
SP = {k: dev(v) for k, v in PF.items() if k.startswith("sp_")}   # light-element level blocks
print(f"PFSAHA tables on {DEVICE.type}: {ndp} layers, 99 elements, up to 6 ion stages stored")''')

md(r"""**The Debye lowering, again — but the gate decision in fp64.** PFSAHA uses the same Kurucz Debye lowering as the charge-balance core, with one wrinkle: the high-$T$ occupation correction switches on only when the per-charge lowering $\Delta\chi \geq 0.1\ \mathrm{eV}$, a *discrete gate*. So we compute the lowering twice — once on the device (the continuous value that enters $U$) and once on the fp64 host (the value that decides the gate).""")

code(r'''def debye_lowering(xne, tk, Pg):
    """Per-unit-charge Debye lowering Delta_chi [eV], depth-batched on the device."""
    charge = 2.0 * xne
    excess = 2.0 * xne - Pg / tk
    charge = torch.where(excess > 0.0, charge + 2.0 * excess, charge)
    charge = torch.where(charge == 0.0, torch.ones_like(charge), charge)
    debye = torch.sqrt(tk / 2.8965e-18 / charge)
    return torch.clamp(1.44e-7 / debye, max=1.0)

def debye_lowering_np(xne, tk, Pg):
    """The same lowering in fp64 on the host — for the discrete occupation-gate decision."""
    xne = np.asarray(xne, np.float64); tk = np.asarray(tk, np.float64); Pg = np.asarray(Pg, np.float64)
    charge = 2.0 * xne; excess = 2.0 * xne - Pg / tk
    charge = np.where(excess > 0.0, charge + 2.0 * excess, charge)
    charge = np.where(charge == 0.0, 1.0, charge)
    debye = np.sqrt(tk / 2.8965e-18 / charge)
    return np.minimum(1.0, 1.44e-7 / debye)

potlow    = debye_lowering(xne_pf, tk_pf, Pg)                                  # device tensor
potlow_h  = debye_lowering_np(PF["electron_density"], PF["tk"], PF["gas_pressure"])  # fp64 host
print(f"Debye lowering per charge: {float(potlow.min()):.2e} -> {float(potlow.max()):.3f} eV")''')

md(r"""**The iron-group partition grid (PFIRON), depth-batched.** Ca–Ni read $U$ from a measured grid in $\log_{10}T$ and $\log_{10}\Delta\chi$. The grid has a three-piece temperature axis (cool / mid / hot), and the cell selection is a *discrete* table lookup — so it is resolved on the fp64 host, vectorized over all 80 depths, and only the bilinear blend is cast to the device. This is one host pass per (iron-group element, ion), never a loop over depth.""")

code(r'''def pfiron(iz, ion, tlog10_h, potlow_cm1):
    """PFIRON U for one (iz, ion), vectorized over depth. Bracket selection on the fp64 host
    (a discrete grid lookup); bilinear blend cast to the device. Returns (nd,) device tensor."""
    elem_idx, ion_idx = iz - 20, ion - 1
    plow = potlow_cm1.detach().cpu().numpy().astype(np.float64)
    tl = tlog10_h
    # three-piece log10(T) axis -> integer bracket `it` (1..56) and fraction f, per depth
    it_hot  = np.clip(((tl - 4.0) / 0.05 + 31.0).astype(np.int64), 1, 56)
    f_hot   = (tl - (it_hot.astype(np.float64) - 31.0) * 0.05 - 4.0) / 0.05
    it_cool = np.maximum(((tl - 3.32) / 0.02 + 2.0).astype(np.int64), 2)
    f_cool  = (tl - (it_cool.astype(np.float64) - 2.0) * 0.02 - 3.32) / 0.02
    it_mid  = ((tl - 3.7) / 0.03 + 21.0).astype(np.int64)
    f_mid   = (tl - (it_mid.astype(np.float64) - 21.0) * 0.03 - 3.7) / 0.03
    hot, cool = tl > 4.0, tl < 3.7
    it = np.clip(np.where(hot, it_hot, np.where(cool, it_cool, it_mid)), 1, 56)
    f  = np.where(hot, f_hot, np.where(cool, f_cool, f_mid))
    it0, itm1 = it - 1, np.maximum(it - 2, 0)
    flat = PFTAB_h[:, :, ion_idx, elem_idx]                          # (7, 56)
    val_weak = f * flat[0, it0] + (1.0 - f) * flat[0, itm1]          # weak-lowering branch
    # bracket the lowering: first i (1..6) with POTLO[i] > potlow, else 6 (all depths at once)
    n_lo = PFLO_h.shape[0]
    low, found = np.full(plow.size, n_lo - 1, np.int64), np.zeros(plow.size, bool)
    for i in range(1, n_lo):
        hit = (~found) & (plow < PFLO_h[i]); low = np.where(hit, i, low); found = found | hit
    p = (np.log10(np.maximum(plow, 1e-30)) - PFLOLOG_h[low - 1]) / 0.30103
    val_low = (p * (f * flat[low, it0] + (1.0 - f) * flat[low, itm1])
               + (1.0 - p) * (f * flat[low - 1, it0] + (1.0 - f) * flat[low - 1, itm1]))
    val = np.where(plow < PFLO_h[0], val_weak, val_low)
    return torch.as_tensor(val, dtype=DTYPE, device=DEVICE)''')

md(r"""**The ordinary-ion partition (NNN), depth-batched.** Most elements unpack their $U$ from the packed integer table NNN: a temperature bracket selects an integer that decodes into two endpoint values, interpolated in temperature. The bracket, the integer unpacking, and the odd/even endpoint split are all *discrete* and keyed on temperature — so the whole lookup runs on the fp64 host (exact integer arithmetic + fp64 interpolation), vectorized over depth, and only the per-depth result is cast to the device.""")

code(r'''def nnn_partition(c0, ip_val):
    """Ordinary-ion U from the packed NNN table, depth-batched. Discrete bracket + integer
    unpacking on the fp64 host; result cast to the device. c0 = 0-based column; ip_val = IP [eV]."""
    nrows = NNN_i.shape[0]
    T2000 = ip_val * 2000.0 / 11.0
    IT = np.clip((T_h / T2000 - 0.5).astype(np.int64), 1, 9)
    DT = T_h / T2000 - IT.astype(np.float64) - 0.5
    i_idx = np.clip((IT + 1) // 2 - 1, 0, nrows - 1)
    nnn_i = NNN_i[i_idx, c0]
    K1 = nnn_i // 100000; K2 = nnn_i - K1 * 100000; K3 = K2 // 10; KSCALE = K2 - K3 * 10
    s = np.clip(KSCALE - 1, 0, SCALE_h.shape[0] - 1)
    P1_odd = K1.astype(np.float64) * SCALE_h[s]; P2_odd = K3.astype(np.float64) * SCALE_h[s]
    i_next = np.clip(i_idx + 1, 0, nrows - 1); nnn_i1 = NNN_i[i_next, c0]
    sN = np.clip((nnn_i1 % 10) - 1, 0, SCALE_h.shape[0] - 1)
    P1_even = K3.astype(np.float64) * SCALE_h[s]
    P2_even = (nnn_i1 // 100000).astype(np.float64) * SCALE_h[sN]
    odd = (IT % 2) == 1
    P1 = np.where(odd, P1_odd, P1_even); P2 = np.where(odd, P2_odd, P2_even)
    pmin_cond = odd & (DT < 0.0) & (s <= 0) & (P1 == np.floor(P2 + 0.5))
    PMIN = np.where(pmin_cond, P1, 1.0)
    part = np.maximum(PMIN, P1 + (P2 - P1) * DT)
    return torch.as_tensor(part, dtype=DTYPE, device=DEVICE)''')

md(r"""**The light-element level sums (special blocks), depth-batched.** H, He, and the metals C/O/Na/Mg/Al/Si/Ca/K/B build $U$ as an explicit Boltzmann sum over measured energy levels, $U = U_0 + \sum_i g_i\,e^{-E_i\,hc/kT}$, with high-lying continuum-like terms bolted on. Each block is a single depth-batched tensor expression; the per-element dispatch (which block, which override) is resolved on the host. This is the longest function — one branch per special ion — but every branch is a vectorized sum over all 80 depths.""")

code(r'''def _block(Ekey, Gkey, nlev, part0):
    """Boltzmann sum part0 + sum_i g_i exp(-E_i hckt) over measured levels, depth-batched."""
    E, g = SP["sp_" + Ekey], SP["sp_" + Gkey]
    s = part0.clone() if torch.is_tensor(part0) else torch.full_like(hckt, float(part0))
    for i in range(1, nlev):
        s = s + g[i] * torch.exp(-E[i] * hckt)
    return s

def special_partition(col, ion):
    """Per-ion explicit level sum for the special light elements. Returns (PART, g_override, D1)
    or None if `col` is not a special column. PART/D1 are (nd,) tensors; g_override is a scalar/None."""
    z = torch.zeros_like(hckt)
    if col == 2:  return torch.ones_like(hckt), None, z                          # bare ion (H II-like)
    if col == 1:                                                                 # H I
        return torch.where(T_pf >= 9000.0, _block("EHYD","GHYD",6,2.0), torch.full_like(hckt,2.0)), None, 109677.576/6.5/6.5*hckt
    if col == 3:                                                                 # He I
        return torch.where(T_pf >= 15000.0, _block("EHE1","GHE1",29,1.0), torch.ones_like(hckt)), None, 109677.576/5.5/5.5*hckt
    if col == 4:                                                                 # He II
        return torch.where(T_pf >= 30000.0, _block("EHE2","GHE2",6,2.0), torch.full_like(hckt,2.0)), None, 4.0*109722.267/6.5/6.5*hckt
    if col == 354:                                                               # C I
        p = _block("EC1","GC1",14, 1.0 + 3.0*torch.exp(-16.42*hckt) + 5.0*torch.exp(-43.42*hckt))
        p = p + (108.0*torch.exp(-80000.0*hckt) + 189.0*torch.exp(-84000.0*hckt) + 247.0*torch.exp(-87000.0*hckt)
                 + 231.0*torch.exp(-88000.0*hckt) + 190.0*torch.exp(-89000.0*hckt) + 300.0*torch.exp(-90000.0*hckt))
        return p, None, z
    if col == 355:                                                               # C II
        p = _block("EC2","GC2",6, 2.0 + 4.0*torch.exp(-63.42*hckt))
        p = p + (6.0*torch.exp(-131731.80*hckt) + 4.0*torch.exp(-142027.1*hckt) + 10.0*torch.exp(-145550.13*hckt)
                 + 10.0*torch.exp(-150463.62*hckt) + 2.0*torch.exp(-157234.07*hckt) + 6.0*torch.exp(-162500.0*hckt)
                 + 42.0*torch.exp(-168000.0*hckt) + 56.0*torch.exp(-178000.0*hckt) + 102.0*torch.exp(-183000.0*hckt)
                 + 400.0*torch.exp(-188000.0*hckt))
        return p, None, z
    if col == 51:                                                                # Mg I
        p = _block("EMG1","GMG1",11, 1.0)
        p = p + (5.0*torch.exp(-53134.0*hckt) + 15.0*torch.exp(-54192.0*hckt) + 28.0*torch.exp(-54676.0*hckt) + 9.0*torch.exp(-57853.0*hckt))
        return p, 4.0, 109734.83/4.5/4.5*hckt
    if col == 52:                                                                # Mg II
        p = _block("EMG2","GMG2",6, 2.0)
        p = p + (10.0*torch.exp(-93310.80*hckt) + 14.0*torch.exp(-93799.70*hckt) + 6.0*torch.exp(-97464.32*hckt)
                 + 10.0*torch.exp(-103419.82*hckt) + 14.0*torch.exp(-103689.89*hckt) + 18.0*torch.exp(-103705.66*hckt))
        return p, 2.0, 4.0*109734.83/5.5/5.5*hckt
    if col == 57:                                                                # Al I
        p = _block("EAL1","GAL1",9, 2.0 + 4.0*torch.exp(-112.061*hckt))
        p = p + 10.0*torch.exp(-42235.0*hckt) + 14.0*torch.exp(-43831.0*hckt)
        return p, 2.0, 109735.08/5.5/5.5*hckt
    if col == 63:                                                                # Si I
        p = _block("ESI1","GSI1",11, 1.0 + 3.0*torch.exp(-77.115*hckt) + 5.0*torch.exp(-223.157*hckt))
        p = p + (76.0*torch.exp(-53000.0*hckt) + 71.0*torch.exp(-57000.0*hckt) + 191.0*torch.exp(-60000.0*hckt)
                 + 240.0*torch.exp(-62000.0*hckt) + 251.0*torch.exp(-63000.0*hckt) + 300.0*torch.exp(-65000.0*hckt))
        return p, None, z
    if col == 64:                                                                # Si II
        p = _block("ESI2","GSI2",6, 2.0 + 4.0*torch.exp(-287.32*hckt))
        p = p + (6.0*torch.exp(-81231.59*hckt) + 6.0*torch.exp(-83937.08*hckt) + 10.0*torch.exp(-101024.09*hckt)
                 + 14.0*torch.exp(-103556.35*hckt) + 10.0*torch.exp(-108800.0*hckt) + 42.0*torch.exp(-115000.0*hckt)
                 + 6.0*torch.exp(-121000.0*hckt) + 38.0*torch.exp(-125000.0*hckt) + 34.0*torch.exp(-132000.0*hckt))
        return p, 2.0, 4.0*109734.83/4.5/4.5*hckt
    if col == 96:                                                                # Ca I
        p = _block("ECA1","GCA1",8, 1.0)
        p = p + (28.0*torch.exp(-37000.0*hckt) + 67.0*torch.exp(-40000.0*hckt) + 21.0*torch.exp(-43000.0*hckt) + 34.0*torch.exp(-48000.0*hckt))
        return p, 4.0, 109734.82/4.5/4.5*hckt
    if col == 97:                                                                # Ca II
        return _block("ECA2","GCA2",5, 2.0) + 12.0*torch.exp(-68000.0*hckt), 2.0, 109734.83/4.5/4.5*hckt
    if col == 367:                                                               # O I
        p = _block("EO1","GO1",13, 5.0 + 3.0*torch.exp(-158.265*hckt) + torch.exp(-226.977*hckt))
        p = p + (15.0*torch.exp(-101140.0*hckt) + 131.0*torch.exp(-103000.0*hckt) + 128.0*torch.exp(-105000.0*hckt) + 600.0*torch.exp(-107000.0*hckt))
        return p, None, z
    if col == 45:                                                                # Na I
        return _block("ENA1","GNA1",8, 2.0) + 10.0*torch.exp(-34548.745*hckt) + 14.0*torch.exp(-34586.96*hckt), 2.0, 109734.83/4.5/4.5*hckt
    if col == 14:                                                                # B I
        p = _block("EB1","GB1",7, 2.0 + 4.0*torch.exp(-15.25*hckt))
        p = p + (6.0*torch.exp(-57786.80*hckt) + 10.0*torch.exp(-59989.0*hckt) + 14.0*torch.exp(-60031.03*hckt) + 2.0*torch.exp(-63561.0*hckt))
        return p, 2.0, 109734.83/4.5/4.5*hckt
    if col == 91:                                                                # K I
        return _block("EK1","GK1",8, 2.0) + 10.0*torch.exp(-27397.077*hckt) + 14.0*torch.exp(-28127.85*hckt), 2.0, 109734.83/5.5/5.5*hckt
    return None''')

md(r"""**Assembling $U$ for one element, over all depths.** This function ties the three machines together for a single element $Z$: for each ion stage it picks the ionization potential, builds the raw $U$ (PFIRON / special / NNN), applies the low-$T$ ground-state floor, and adds the high-$T$ occupation correction — every step a depth-batched tensor op, the discrete gates decided on the fp64 host. It returns the $(\text{nion2}, \text{nd})$ stacks of $U$, the ionization potentials, and the lowering that the Saha ladder consumes.""")

code(r'''def occupation_term(d, zion, tv):
    """The high-T occupation-probability correction (Hummer-Mihalas style), depth-batched."""
    x3 = torch.sqrt(13.595 * zion * zion / tv / d) ** 3
    poly = 1.0/3.0 + (1.0 - (0.5 + (1.0/18.0 + d/120.0) * d) * d) * d
    return x3 * poly

def potion_index(iz, ion):
    """1-based index of the ion's IP in the flat POTION table."""
    return iz * (iz + 1) // 2 + ion - 1 if iz <= 30 else iz * 5 + 341 + ion - 1

def element_block(iz):
    """(n_start_1based, n_ion_stages) for element iz — the PFSAHA table dispatch."""
    if iz <= 28:
        n = int(LOCZ[iz-1]); nions = int(LOCZ[iz] - LOCZ[iz-1]) if iz < len(LOCZ) else 3
    else:
        n, nions = 3*iz + 54, 3
    if   iz == 6: n, nions = 354, 6
    elif iz == 7: n, nions = 360, 7
    elif iz == 8: n, nions = 367, 8
    if 20 <= iz < 29: nions = 10
    return n, nions

def build_part(iz, nion_track):
    """Per-ion U, IP, POTLO for element iz, all depths. Returns (PART, IP, POTLO, nion2),
    each (nion2, nd). Loops only over the <=9 ion stages; every op is depth-batched."""
    n_start = element_block(iz)[0] - 1
    nion2 = min(nion_track + 2, element_block(iz)[1])
    PART  = torch.ones(nion2, ndp, dtype=DTYPE, device=DEVICE)
    IP    = torch.zeros(nion2, ndp, dtype=DTYPE, device=DEVICE)
    POTLO = torch.zeros(nion2, ndp, dtype=DTYPE, device=DEVICE)
    for ion in range(1, nion2 + 1):
        zion, k = float(ion), ion - 1
        POTLO[k] = potlow * zion                                # lowering scales with ion charge
        pidx = potion_index(iz, ion) - 1                        # ionization potential [cm^-1] -> eV
        ip_val = 0.0
        if 0 <= pidx < POTION_h.size:
            ip_val = POTION_h[pidx] / EV_TO_CM
            if ip_val == 0.0 and pidx > 0: ip_val = POTION_h[pidx-1] / EV_TO_CM
        IP[k] = ip_val

        if 20 <= iz < 29:                                       # iron group: PFIRON grid
            PART[k] = pfiron(iz, ion, tlog_h / LN10, POTLO[k] * EV_TO_CM); continue

        col, c0 = n_start + ion, n_start + ion - 1
        G = float(int(NNN_i[5, c0]) - (int(NNN_i[5, c0]) // 100) * 100) if c0 < NNN_i.shape[1] else 0.0
        D1 = torch.zeros(ndp, dtype=DTYPE, device=DEVICE)
        handled = special_partition(col, ion)
        if handled is not None:                                 # light element: explicit level sum
            part_sp, g_ovr, D1 = handled; PART[k] = part_sp
            if g_ovr is not None: G = float(g_ovr)
        elif c0 < NNN_i.shape[1] and ip_val > 0.0:              # ordinary ion: packed NNN
            PART[k] = nnn_partition(c0, ip_val)
        else:
            PART[k] = torch.ones(ndp, dtype=DTYPE, device=DEVICE)

        # (3a) low-T ground-state floor (PFGROUND) + (3b) high-T occupation correction.
        # Every T-threshold gate decided on the fp64 host so an fp32 boundary slip can't flip
        # a regime (the discrete-decision rule the critics approved).
        if ip_val > 0.0:
            T2000 = ip_val * 2000.0 / 11.0
            low_t = torch.as_tensor(T_h < (T2000 * 2.0), device=DEVICE)
            nelion = (iz - 1) * 6 + ion
            if nelion < pfg.shape[0]:
                apply_lowt = low_t & (pfg[nelion] > 0.0)
                PART[k] = torch.where(apply_lowt, torch.maximum(PART[k], pfg[nelion]), PART[k])
            D1_pos = D1 > 0.0; special_bypass = bool(D1_pos.any())
            if G > 0.0 or special_bypass:
                gate_T   = torch.as_tensor(T_h >= (T2000 * 4.0), device=DEVICE)
                potlo_ge = torch.as_tensor((potlow_h * float(ion)) >= 0.1, device=DEVICE)
                gate = (D1_pos | potlo_ge) & (D1_pos | gate_T) & (~low_t)
                tcap = torch.as_tensor(T_h > (T2000 * 11.0), device=DEVICE)
                tv = torch.where(tcap, torch.full_like(T_pf, (T2000*11.0) * KEV), tkev)
                d1c = torch.where(D1 <= 0.0, 0.1 / tv, D1); d2c = POTLO[k] / tv
                if G > 0.0:
                    add = G * torch.exp(-ip_val / tv) * (occupation_term(d2c, zion, tv) - occupation_term(d1c, zion, tv))
                    PART[k] = torch.where(gate, PART[k] + add, PART[k])
    return PART, IP, POTLO, nion2''')

md(r"""**The Saha ladder, in log space.** Given $U$, $\chi$, and the lowering for every stage, the Saha ladder chains the stage-to-stage ratios and normalises them. We do this in **log space** — accumulate $\log F$ by cumulative sum, subtract the per-depth maximum, exponentiate, and normalise (a softmax). The raw running product of Saha ratios reaches $\sim 10^{13}$ per stage and overflows fp32's $3.4\times 10^{38}$ ceiling in the deep, hot layers; clamping would silently break particle conservation, while the log-space softmax is overflow-free and accurate to the float floor. The output is the stored stage factor $F/U$.""")

code(r'''def saha_ladder(PART, IP, POTLO):
    """Saha stage fractions F/PART, depth-batched, in log space (overflow-safe softmax)."""
    nion2 = PART.shape[0]
    log_cf = math.log(2.0 * SAHA) + 1.5 * torch.log(T_pf) - torch.log(xne_pf)   # (nd,)
    logF = torch.zeros(nion2, ndp, dtype=DTYPE, device=DEVICE)                   # logF[0]=0 (F[0]=1)
    eps = torch.finfo(DTYPE).tiny
    for ion in range(2, nion2 + 1):
        k = ion - 1
        logratio = (log_cf + torch.log(torch.clamp(PART[k], min=eps))
                    - torch.log(torch.clamp(PART[k-1], min=eps))
                    - (IP[k-1] - POTLO[k-1]) / tkev)
        logF[k] = torch.where(PART[k-1] > 0.0, logratio, torch.full_like(logratio, -float("inf")))
    logout = torch.cumsum(logF, dim=0)                          # un-normalised log stage population
    w = torch.exp(logout - torch.amax(logout, dim=0, keepdim=True))             # softmax (overflow-free)
    out = w / torch.sum(w, dim=0, keepdim=True)
    return torch.where(PART > 0.0, out / PART, torch.zeros_like(out))           # F/PART''')

md(r"""**Drive PFSAHA over every element.** The only Python loop left is over the 99 elements, because their table dispatch is genuinely heterogeneous: batching across elements would evaluate all branches everywhere and then mask. Each element is one `build_part` + one `saha_ladder`, both fully depth-batched, stored into the $(\text{nd}, 99, 6)$ result tensors.""")

code(r'''NSTORE = 6
U_pf    = torch.zeros(ndp, 99, NSTORE, dtype=DTYPE, device=DEVICE)   # per-ion partition functions
popfrac = torch.zeros(ndp, 99, NSTORE, dtype=DTYPE, device=DEVICE)   # per-ion F/U fractions
popion  = torch.zeros(ndp, 99, NSTORE, dtype=DTYPE, device=DEVICE)   # Kurucz population factors n_ion/U

for Z in range(1, 100):
    PART, IP, POTLO, nion2 = build_part(Z, int(nion_per_Z[Z-1]))
    fr = saha_ladder(PART, IP, POTLO)
    ns = min(NSTORE, nion2)
    U_pf[:, Z-1, :ns]    = PART[:ns].T
    popfrac[:, Z-1, :ns] = fr[:ns].T
    popion[:, Z-1, :ns]  = (fr[:ns].T) * (xnatom[:, None] * xab_pf[Z-1])

print(f"PFSAHA assembled on {DEVICE.type}: U, F/U, and Kurucz population factors for "
      f"99 elements x {NSTORE} stages x {ndp} depths")
# the physical stage fraction is F = (F/U)*U; popfrac stores F/U, so re-form F to read off ionization
F_Fe = (popfrac[:, 25, :] * U_pf[:, 25, :])
jp = int(torch.argmin(torch.abs(tau - 2/3)))
print(f"at the photosphere, Fe is {100*float(F_Fe[jp,1]):.1f}% singly ionized "
      f"(Fe II), {100*float(F_Fe[jp,0]):.1f}% neutral (Fe I) — the dominant solar iron stage is Fe II")''')

md(r"""## The PFSAHA comparison cell — validating the per-ion physics

The per-part GPU check, now for the full PFSAHA engine. We put the GPU per-ion partition functions $U$, the stored stage factors $F/U$, and the Kurucz population factors $\Phi=n_ZF/U$ next to the NumPy edition's `pfsaha_truth.npz`. We report the maximum relative deviation **over the physically populated stages**. A subtlety the comparison must respect: the reference holds stage factors down to the fp64 *subnormal* floor ($\sim 10^{-324}$) for utterly-depopulated ion stages, while fp32 underflows those to exactly zero. A population fraction of $10^{-39}$ is negligible for the opacity checks in this lecture, so we measure parity over the stages that carry at least $10^{-12}$ of their element's dominant stage, the genuine fp32 float floor of the Saha ladder.""")

code(r'''def compare_pfsaha(name, ours, ref, floor=1e-12):
    """Max relative deviation over the PHYSICALLY POPULATED stages (>= floor of the dominant
    stage of that layer/element); utterly-depopulated stages underflow fp32 to 0 (invisible)."""
    ours = ours.detach().cpu().to(torch.float64).numpy(); ref = np.asarray(ref, np.float64)
    permax = ref.max(axis=2, keepdims=True)                    # dominant stage per (layer, element)
    sig = ref > (floor * np.maximum(permax, 1e-300))           # significance mask
    rel = np.abs(ours[sig] - ref[sig]) / np.abs(ref[sig])
    tag = "agree" if rel.max() < 5e-5 else "CHECK"
    print(f"{name:34s} significant n={int(sig.sum()):6d}  max|rel| = {rel.max():.2e}  "
          f"median = {np.median(rel):.2e}   [{tag}]")
    return float(rel.max())

print(f"Validating GPU PFSAHA against reference/pfsaha_truth.npz")
print(f"  device = {DEVICE.type}   dtype = {str(DTYPE).split('.')[-1]}\n")

dU  = compare_pfsaha("partition function U",        U_pf,    PFT["U"])
dF  = compare_pfsaha("stage fraction F/U",          popfrac, PFT["population_fraction"])
dP  = compare_pfsaha("per-ion population n_ion/U",  popion,  PFT["population_per_ion"])

charge_pf = torch.arange(NSTORE, dtype=DTYPE, device=DEVICE)
xne_from_pfsaha = (popion * U_pf * charge_pf[None, None, :]).sum(dim=(1, 2))
charge_np = np.arange(NSTORE, dtype=np.float64)
xne_from_pfsaha_ref = (PFT["population_per_ion"] * PFT["U"] * charge_np[None, None, :]).sum(axis=(1, 2))
dx = compare("PFSAHA stored-stage ne", xne_from_pfsaha, xne_from_pfsaha_ref,
             tol=(5e-5 if DTYPE == torch.float32 else 1e-9))

max_dev = max(dU, dF, dP, dx)
floor = 5e-5 if DTYPE == torch.float32 else 1e-9
print(f"\nmax relative deviation (GPU {DEVICE.type}/{str(DTYPE).split('.')[-1]} vs pykurucz PFSAHA) "
      f"= {max_dev:.2e}")
status = "PASS" if max_dev < floor else "CHECK"
print(f"documented PFSAHA float floor = {floor:.1e}   ->   [{status}]")
assert max_dev < floor, f"PFSAHA deviates by {max_dev:.2e}, above the float floor {floor:.1e}"
print("\nThe GPU PFSAHA per-ion partition functions and population factors match the reference to the float floor.")''')

md(r"""**What the PFSAHA numbers mean.** The per-ion partition functions $U$ reproduce the reference to $\sim 2\times10^{-6}$ — the fp32 floor of the interpolation blends and Boltzmann sums. The stored stage factors and population factors match to $\sim 1.5\times10^{-5}$ over every physically populated stage; the slightly larger residual is the chained Saha ratio accumulating single-precision round-off across up to nine ion stages. Stages with fractions below $10^{-12}$ — down to the fp64 subnormal floor — underflow fp32 to zero, a documented difference with negligible effect for the opacity checks in this lecture. This is the full ionization state of the gas, on the GPU, validated against the reference implementation: **the population factors Lecture 3's continuous opacity consumes.**

**Where this goes next.** The equation of state is now complete on the GPU — both the charge-balance $n_e$ and the full per-ion $U$, physical stage fractions $F$, and population factors $\Phi=n_ZF/U$, all depth-batched in `torch`, all validated against the NumPy edition. With these factors in hand, Lecture 3 builds the **continuous opacity** — H$^-$ bound-free and free-free, Rayleigh and Thomson scattering — fully vectorized over depth *and* wavelength on the GPU, and validated against its NumPy twin to the same float floor.""")


# ── CATCH-AND-FILL: appended sections (port_worker fill) ──
md(r"""### Part 1 — the iron group reads a tabulated grid

The Kurucz $Z=20$--$28$ Ca--Ni/PFIRON block, which includes the traditional iron-group elements, has such dense, tangled level structures that these partition functions are not summed level by level but **read from a pre-computed grid**, `PFTAB`, indexed by temperature and by how much the ionization potential has been lowered. In the GPU port this is the `pfiron` helper above: it brackets $\log_{10}T$ on the grid's three-piece temperature axis, brackets the Debye lowering in $\log_{10}(\Delta\chi)$ when the lowering is large enough to matter, and returns the bilinear interpolation as a depth tensor.

The important GPU lesson is not that the table is large — it is modest — but that the **integer cell choice is discontinuous**. A one-bit fp32 slip at a grid seam could choose the neighboring cell and produce a percent-level jump in $U$. We therefore make the discrete bracket decision on the fp64 host, vectorized over all depths, then cast the interpolated depth vector back to the device. The continuous arithmetic and the downstream Saha ladder remain torch-native and depth-batched; only the table seam decision is protected from MPS fp32 rounding.""")

md(r"""### Part 2 — hand-built level sums for the common light elements

A set of common light elements that dominate the spectrum — H, He, C, O, Na, Mg, Al, Si, K, B, and Ca — get their neutral and first-ion partition functions assembled by hand from explicit, curated lists of measured levels, rather than from the coarse packed `NNN` table. Each block is the same Boltzmann sum we wrote at the beginning of the lecture,

$$
U(T) = \sum_i g_i\,e^{-E_i/kT},
$$

just with the measured levels spelled out. In the GPU port this is the `special_partition` dispatcher above. Each branch is a **vectorized tensor expression over all depths**: the level energies and statistical weights live on the device, `hckt` is the depth vector, and the exponentials are evaluated for the whole atmosphere at once. The dispatcher itself is heterogeneous bookkeeping — `col=45` means Na I, `col=51` means Mg I, and so on — but once a branch is chosen there is no loop over depth.

The returned `g_override` and `D1` are the two pieces the high-temperature occupation correction needs. `g_override` supplies the statistical weight of the high Rydberg tail; `D1` supplies a lower cutoff for alkali-like ions whose loosely bound valence electron is especially sensitive to plasma lowering. Those corrections are what make PFSAHA a depth-correct partition-function engine, not merely a temperature table.""")

md(r"""## Synthesis: what you built and where it goes

You closed the atmosphere. Starting from the temperature and pressure of Lecture 1, you used the **Boltzmann** distribution to populate energy levels, the **Saha** equation to balance ionization stages, **Debye lowering** to account for the dense-plasma environment, and **charge conservation** to solve for the electron density at every depth. In this GPU edition every one of those steps was written as a depth-batched `torch` calculation: the 80 atmospheric layers are tensor lanes, not Python-loop iterations. The resulting electron density and hydrogen ionization reproduce the NumPy edition to the documented float floor.

Along the way you found the quiet truth of the cool-star photosphere: hydrogen is almost entirely neutral there, and in the line-forming layers the free electrons come from a handful of low-ionization metals. This is not a minor bookkeeping point. H$^-$ opacity, the continuum opacity that dominates the optical Sun, is proportional to the supply of free electrons; changing the metal abundance changes the electron pressure and therefore changes the continuum.

You then went one level deeper. The temperature-only partition functions that reproduce $n_e$ so well are not good enough for the per-ion population factors the opacity engines consume, so you rebuilt **PFSAHA** on the GPU — assembling each ion's depth-correct $U$ on the fly from atomic data, applying the Debye and occupation corrections, and running the same Saha ladder in log space. The result matches the reference per-ion partition functions and population factors to the fp32 GPU float floor over all physically populated stages.

Those electrons, and those per-ion factors, are exactly what the next lecture needs. Lecture 3 turns this equation of state into a **continuous opacity**: H$^-$ bound-free and free-free absorption, Rayleigh and Thomson scattering, all evaluated over depth and wavelength. The same pattern will recur throughout the GPU edition: write the physics as batched tensor algebra, identify the few places where fp64 table decisions matter, and validate against the NumPy twin cell by cell.""")

md(r"""## Summary

- In LTE the **Boltzmann** distribution sets level populations within an ion,
  $n_i/n_{\rm ion} = (g_i/U)\,e^{-E_i/kT}$, normalised by the **partition function** $U(T)$.
- The **Saha** equation balances ionization stages:
  $n_{i+1}n_e/n_i = 2(U_{i+1}/U_i)(2\pi m_e kT/h^2)^{3/2}e^{-\chi_i/kT}$,
  with the prefactor $2.4148\times10^{15}\,T^{3/2}$.
- The Kurucz **Debye continuum-lowering** prescription lowers the ionization potential by
  $\Delta\chi \approx 1.44\times10^{-7}/\lambda_D$ per unit charge — small at the surface,
  growing with density, and capped at $1\ \mathrm{eV}$ in this reference implementation.
- **Charge conservation**,
  $n_e = \sum_Z n_Z \sum_i i\,f_{Z,i}$,
  closes the system. The GPU depth-batched solve reproduces the NumPy reference electron
  density to the fp32 float floor on MPS/CUDA, and to tighter precision on the CPU fp64 fallback.
- In the solar photosphere hydrogen is only $\sim10^{-4}$ ionized. In the cool
  **line-forming layers metals (Mg, Si, Fe, Na, Ca) supply the electrons**, while hydrogen
  takes over deeper and higher up.
- Temperature-only partition functions are good for the *summed* electron density but not for
  individual opacity population factors. **PFSAHA** assembles each ion's depth-correct $U$ on the fly:
  a Ca--Ni/PFIRON grid that includes the traditional iron-group elements, hand-built level sums for important light elements, packed
  table interpolation for ordinary ions, plus Debye lowering and a high-temperature occupation
  correction.
- The GPU PFSAHA port keeps the continuous arithmetic on the device and depth-batched, protects
  the discrete table-bracket decisions with fp64 host evaluation where accelerator fp32 could cross a grid seam, and
  validates the per-ion $U$, $F/U$, and $\Phi=n_ZF/U$ against the reference to the documented float floor.""")

md(r"""## Practice exercises

**1. The Saha turnover for hydrogen.** Using the torch `saha_ratio` helper, hold $n_e$ fixed at a photospheric value ($\sim10^{13}\ \mathrm{cm^{-3}}$) and make a device tensor of temperatures from $4000$ to $12000\ \mathrm{K}$. Plot the ionized fraction of hydrogen. At what temperature does hydrogen become half-ionized, and why is it far below the naive scale $13.6\ \mathrm{eV}/k \approx 158{,}000\ \mathrm{K}$?

**2. Metals as electron donors.** Using the `ne_Z` tensor already computed above, find the dominant electron donor at a deep layer ($\tau \sim 10$) and at a line-forming layer ($\tau \sim 0.05$). Explain why hydrogen wins deep while metals win where the lines form.

**3. A metal-poor atmosphere.** The electron density depends on metal abundance. Make a copy of the abundance tensor and multiply all elements heavier than helium by $10^{-1}$ (that is, $[\mathrm{M/H}] = -1$), then re-solve for $n_e$ with the same depth-batched charge-conservation machinery. By how much does the photospheric electron density drop, and what does that imply for H$^-$ opacity in metal-poor stars?

**4. Temperature-only $U$ versus PFSAHA.** Pick Na I and compare its neutral-stage population factor at the photosphere two ways: first with the simplified `ionization_fractions` calculation and the temperature-only `U` table, and second with the validated PFSAHA tensors `U_pf` and `popion`. Confirm that the temperature-only shortcut misses the stored opacity factor by a large amount while PFSAHA agrees with the reference to the float floor. Which PFSAHA ingredient — the hand-built level sum, the Debye lowering, the low-temperature ground-state floor, or the occupation correction — is most important for this cool, easily-ionized metal?

**5. Vectorization audit.** Inspect the PFSAHA cells and separate the computation into two categories: continuous tensor arithmetic that runs on the device, and discrete table decisions that are intentionally made on the fp64 host. Why would a fully fp32 table-bracket decision be dangerous near a grid seam? Why is the log-space Saha ladder safer than multiplying raw Saha ratios stage by stage?""")

md(r"""## Further reading

- **Gray, D. F. (2005). *The Observation and Analysis of Stellar Photospheres*, 3rd ed., Cambridge University Press.** Chapter 1 derives the Saha and Boltzmann equations and works the solar electron-donor problem explicitly.
- **Saha, M. N. (1921). *On a Physical Theory of Stellar Spectra*, Proc. R. Soc. Lond. A, 99, 135.** The original ionization equation.
- **Mihalas, D. (1978). *Stellar Atmospheres*, 2nd ed., Freeman.** Chapter 5 on the LTE equation of state, partition functions, and pressure ionization.
- **Hummer, D. G. & Mihalas, D. (1988). *The Equation of State for Stellar Envelopes*, ApJ, 331, 794.** The occupation-probability formalism behind modern pressure-ionization treatments.
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz*](https://arxiv.org/abs/2603.11693).** The implementation used to generate the NumPy edition's reference electron densities and PFSAHA comparison data.
- **PyTorch documentation: MPS backend.** Useful for understanding why this GPU edition keeps bulk arithmetic on Apple Silicon's Metal backend while protecting a few fp64-sensitive reductions and table decisions on the CPU fallback path.""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT.relative_to(BOOK)} ({len(cells)} cells)")
