#!/usr/bin/env python
"""Assemble content/Lecture2.ipynb (unexecuted). Execute + render via build.py.

Lecture 2 -- The Equation of State. These are self-contained graduate notes on
Boltzmann level populations, Saha ionization balance, pressure lowering, charge
conservation, and the electron density needed by continuum and line opacity.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture2.ipynb"

cells = []
def md(src): cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

md(r"""# Lecture 2 — The Equation of State

*Self-contained notes on ionization, level populations, and electron density*

*Yuan-Sen Ting*

An atmosphere model gives temperature, pressure, and density as functions of depth. Opacity needs one more layer of information: how many particles are neutral atoms, ions, excited atoms, and free electrons. That missing information is the **equation of state**. It is the bridge from thermodynamic structure to opacity: a line can absorb only if the relevant element is in the right ionization stage and the lower energy level is populated; the optical continuum of a cool star depends strongly on the free-electron density because H-minus, the negative hydrogen ion, needs an extra electron to exist.

The bridge has three pieces. Boltzmann statistics distribute atoms among energy levels inside one ion. The Saha equation distributes atoms among ionization stages. Charge conservation determines the electron density that makes all of those ionization balances mutually consistent. At the end we extend the same idea to per-ion population factors, the quantities that detailed opacity formulas multiply by level weights and Boltzmann factors.""")

md(r"""## Introduction

A stellar atmosphere is not ready for spectroscopy until it knows its particle populations. The dominant continuum opacity of a cool star, H-minus, is a hydrogen atom that has *captured* a free electron, so it depends directly on $n_e$; and most line strengths depend on $n_e$ indirectly, through the ionization balance that decides how many atoms sit in the right stage to absorb. Before we can compute an opacity we must answer two questions at each depth: **which ionization stage is each element in, and how are its electrons distributed among energy levels?**

In local thermodynamic equilibrium the answer is fixed by the temperature, the particle densities, and the chemical abundances, through two classical results: the **Boltzmann** distribution for the populations of energy levels within an ion, and the **Saha** equation for the balance between successive ionization stages. We work in the ideal-gas LTE equation of state and, for this solar example, set molecular equilibria aside. Closing the system requires one more condition — **charge conservation**, that the free electrons exactly balance the positive ions — and that is what finally pins down $n_e$.

The equations are evaluated over the full depth grid at once. PyTorch is used as the array language so the same expressions can run GPU-natively when an accelerator is available.""")

md(r"""## Atmosphere and atomic data

The calculation uses the solar atmosphere and atomic data arrays shipped with the book. The atmosphere supplies the depth grid, temperature, and gas pressure. The atomic data supply partition functions, ionization potentials, and abundances. The equations below compute the ionization balance and electron density from those inputs.""")

code(r'''import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt

# Only execution machinery: the EOS equations below do not depend on this choice.
if torch.backends.mps.is_available():
    DEVICE, DTYPE = torch.device("mps"), torch.float32
elif torch.cuda.is_available():
    DEVICE, DTYPE = torch.device("cuda"), torch.float32
else:
    DEVICE, DTYPE = torch.device("cpu"), torch.float64

def dev(x):
    """Convert an input array to the working array type."""
    return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=DEVICE)

# one shared Matplotlib style for every figure in the lecture
plt.rcParams.update({
    "figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
})''')

md(r"""Load the atmosphere and atomic data. The compact historical keys are translated into readable names as soon as they enter the notebook.""")

code(r'''REF = np.load(pathlib.Path("..") / "reference" / "L2.npz")

# atmosphere columns and atomic data, converted once for the calculations below
tau = dev(REF["tau"]); T = dev(REF["T"]); P_gas = dev(REF["P_gas"]); tk = dev(REF["tk"])  # tk = k*T [erg]
nd = T.shape[0]
print(f"{nd} atmosphere layers; photosphere near tau=2/3 at "
      f"T = {float(T[int(torch.argmin(torch.abs(tau - 2/3)))]):.0f} K")''')

# ── Boltzmann + partition function ──────────────────────────────────────
md(r"""## The Boltzmann distribution and the partition function

Within a single ion, the population of an energy level is set by the **Boltzmann distribution**. If level $i$ has energy $E_i$ above the ground state and statistical weight (degeneracy) $g_i=2J_i+1$, then in LTE the fraction of that ion's atoms in level $i$ is

$$
\frac{n_i}{n_{\rm ion}} = \frac{g_i}{U(T)}\,e^{-E_i/kT},
\qquad
U(T) = \sum_i g_i\,e^{-E_i/kT},
$$

where $n_{\rm ion}$ is the total number density of that one ionization stage and $U(T)$ is the **partition function** — the effective number of accessible states at temperature $T$. Computing $U(T)$ from scratch means summing thousands of measured levels per ion. That is atomic data, not new physics to derive here, so we use tabulated partition functions and focus on the ionization physics that consumes them. The first plot compares neutral hydrogen and neutral iron across the atmosphere.""")

code(r'''# partition functions U[layer, Z-1, ion]; ion=0 is the neutral stage
U = dev(REF["U"])

# host arrays are only for plotting
T_np = T.detach().cpu().numpy()
plt.plot(T_np, U[:, 0, 0].cpu(),  "o-", ms=3, label="H I  (Z=1, neutral)")
plt.plot(T_np, U[:, 25, 0].cpu(), "s-", ms=3, label="Fe I (Z=26, neutral)")
plt.xlabel("temperature  [K]"); plt.ylabel(r"partition function  $U(T)$")
plt.title("Neutral-stage partition functions grow as excited levels switch on")
plt.legend(); plt.tight_layout(); plt.show()

print(f"U(H I) ranges {float(U[:,0,0].min()):.2f}-{float(U[:,0,0].max()):.2f};  "
      f"U(Fe I) ranges {float(U[:,25,0].min()):.1f}-{float(U[:,25,0].max()):.1f}")''')

md(r"""Hydrogen's neutral partition function hugs $2$ — its first excited level sits $10.2\ \mathrm{eV}$ up, out of reach at photospheric temperatures, so only the ground state ($g_0 = 2$) is populated. Formally the hydrogen partition function diverges without truncating the Rydberg series; pressure and occupation physics cut off those high levels, and at photospheric temperatures the result is essentially the ground doublet. Iron, with a forest of low-lying levels from its open $d$-shell, climbs to $\sim 30$ as those levels thermally switch on. These numbers feed straight into the Saha equation next.

![Saha sets the balance between ionization stages; Boltzmann sets the level populations within an ion — together they fix which species can absorb, and in the solar line-forming photosphere the metals are the major electron donors.](resources/figures/s2_saha_boltzmann.png)""")

# ── Saha equation ───────────────────────────────────────────────────────
md(r"""## The Saha equation

Boltzmann distributes atoms among levels *within* an ion; the **Saha equation** distributes them *between* ionization stages:

$$
\frac{n_{i+1}}{n_i}\,n_e
= 2\,\frac{U_{i+1}}{U_i}\,
\left(\frac{2\pi m_e k T}{h^2}\right)^{3/2}
e^{-\chi_i/kT}.
$$

The **$2$** is the freed electron's two spin states; $U_{i+1}/U_i$ compares the two ions' internal states; $(2\pi m_e kT/h^2)^{3/2}$ is the electron's translational phase-space density (the inverse cube of its thermal de Broglie wavelength); $e^{-\chi_i/kT}$ is the Boltzmann penalty for the ionization energy $\chi_i$; and the lone $n_e$ on the left says **the more electrons are already around, the harder it is to stay ionized**.

The thermal-de-Broglie prefactor is a constant, $(2\pi m_e k/h^2)^{3/2} = 2.4148\times10^{15}\ \mathrm{cm^{-3}\,K^{-3/2}}$. The helper below returns $N_{i+1}/N_i$ at every depth. The constants `SAHA` and `KEV` are written explicitly so the units are visible: `SAHA` is the phase-space prefactor, and `KEV` converts kelvin to electron volts.""")

code(r'''SAHA = 2.4148e15        # (2*pi*m_e*k / h^2)^{3/2}  [cm^-3 K^-3/2]
KEV  = 8.6171e-5        # eV per kelvin: kT[eV] = KEV*T, so chi[eV]/(KEV*T) = chi/kT

def saha_ratio(U_lo, U_hi, chi_eV, T, n_e, dchi_eV=0.0):
    """Saha ratio N_{i+1}/N_i. chi is lowered by dchi_eV (Debye, below)."""
    return (2.0 * U_hi / U_lo) * SAHA * T**1.5 * torch.exp(-(chi_eV - dchi_eV) / (KEV * T)) / n_e''')

# ── Debye lowering ──────────────────────────────────────────────────────
md(r"""## Pressure ionization: Debye lowering

In an isolated atom the ionization potential is set by the Coulomb attraction between the electron and the nucleus. In a plasma, nearby ions and electrons partly screen that electric field. The atom therefore behaves as if its ionization potential were slightly smaller. This effect is called **pressure ionization** or **continuum lowering**.

The simple approximation used here estimates the shielding length with the **Debye radius**,

$$
\lambda_D = \sqrt{\frac{kT}{4\pi e^2 n_{\rm charge}}},
$$

where $n_{\rm charge}$ is the density of charged particles that screen the ion. A smaller Debye radius means stronger screening. The corresponding lowering of the ionization potential is

$$
\Delta\chi_i = \frac{i\,e^2}{\lambda_D} \;\approx\; i \times \frac{1.44\times10^{-7}\ \mathrm{eV\,cm}}{\lambda_D},
$$

where $i$ is the charge of the ion after the electron leaves. The transition from a neutral atom to a singly ionized ion therefore uses $i=1$; from singly to doubly ionized uses $i=2$. The prescription caps the lowering at $1\ \mathrm{eV}$ so the approximation cannot run into an unphysical regime in dense layers. Since $\lambda_D$ depends on $n_e$, the lowering is recomputed during the charge-conservation iteration below.""")

code(r'''# Debye lowering per unit charge [eV] from the input atmosphere, shown before solving the loop
potlow = dev(REF["potlow"])
print(f"Debye lowering Delta_chi (per unit charge): "
      f"{float(potlow.min()):.2e} eV (top)  ->  {float(potlow.max()):.3f} eV (deepest layer)")''')

md(r"""**A first read on hydrogen.** Hydrogen is the cleanest Saha example: two stages, H I and the bare proton H II, with ionization potential $\chi = 13.6\ \mathrm{eV}$. To see the size of the effect before solving the full coupled problem, use the electron-density profile supplied with the atmosphere as an input to the Saha ratio. The next section solves that electron density self-consistently from charge conservation.""")

code(r'''chi = dev(REF["chi"])
electron_density_input = dev(REF["xne"])

U_HI, U_HII = U[:, 0, 0], U[:, 0, 1]          # H I and H II partition functions (depth tensors)
chi_H = chi[0, 0]                             # hydrogen ionization potential [eV] (13.6)

# Saha ratio N(H II)/N(H I); Debye lowering at charge 1 (the neutral -> singly transition)
r = saha_ratio(U_HI, U_HII, chi_H, T, electron_density_input, dchi_eV=potlow * 1)
frac_HII = r / (1.0 + r)                       # ionized fraction n(H II)/n(H total)

jphot = int(torch.argmin(torch.abs(tau - 2/3)))
print(f"H is {100*float(frac_HII[jphot]):.4f}% ionized at the photosphere")
print(f"H II fraction spans {float(frac_HII.min()):.2e} -> {float(frac_HII.max()):.2e} through the atmosphere")''')

md(r"""The headline number is striking: near the photosphere hydrogen is only a tiny fraction ionized. The Sun's most abundant element therefore contributes surprisingly few free electrons where much of the visible spectrum forms. That is why the electron density cannot be guessed from hydrogen alone; it has to be solved from all elements together.""")

# ── Charge conservation ─────────────────────────────────────────────────
md(r"""## Charge conservation: solving for the electron density

Within this LTE, ideal-gas-plus-Debye atmosphere, the electrons come from whichever elements ionize most easily — Na, Mg, Al, Si, K, Ca, Fe, with first ionization potentials well below hydrogen's $13.6\ \mathrm{eV}$, roughly $4$--$8\ \mathrm{eV}$, and therefore large singly ionized fractions in the relevant photospheric layers. To find $n_e$ we impose **charge conservation**,

$$
n_e = \sum_{\text{elements } Z}\; n_Z \sum_{i} i\,f_{Z,i},
$$

where $n_Z = A_Z\,n_{\rm atom}$ and $A_Z$ is the number fraction of element $Z$ over all atomic nuclei. The fraction $f_{Z,i}$ is the fraction of element $Z$ in charge stage $i$: $i=0$ is neutral, $i=1$ is singly ionized, and so on. Spectroscopic notation is shifted by one Roman numeral: charge stage $0$ is H I or Fe I, charge stage $1$ is H II or Fe II.

There is a circularity. The Saha fractions need $n_e$, but $n_e$ is the sum of the charges those fractions create. We break the loop by fixed-point iteration: guess $n_e$, compute all ionization fractions, sum the donated charge, damp the update, and repeat until the answer stops changing. The nuclei density in each iteration follows from the ideal-gas relation

$$
P_{\rm gas} = (n_{\rm atom} + n_e) kT.
$$

Some elements need more ion stages for a stable Saha normalization than they need for the electron sum in this temperature range. The arrays `nion2` and `nion` represent those two choices: normalize the ion fractions over the longer ladder, then count charge over the tracked stages. That distinction is small in the photosphere but important in hot layers.""")

code(r'''xab   = dev(REF["xabund"])          # abundance A_Z: number fraction over all atoms (99 elements)
nion  = REF["nion"].astype(int)     # stages whose charge we count per element (host int)
nion2 = REF["nion2"].astype(int)    # ladder length used to normalize the Saha fractions

def ionization_fractions(Z, T, n_e, dchi1):
    """Fractions f[layer, ion] of element Z, chaining Saha ratios over the nion2-stage
    ladder, returned as (depth, ion)."""
    ni2 = int(nion2[Z-1])

    # r[:,0]=1: every stage measured against the neutral one. Climb the ladder one stage
    # at a time; the Debye lowering scales with the charge i of the resulting ion.
    cols = [torch.ones(nd, dtype=DTYPE, device=DEVICE)]
    for i in range(1, ni2):
        cols.append(cols[i-1] * saha_ratio(U[:, Z-1, i-1], U[:, Z-1, i],
                                           chi[Z-1, i-1], T, n_e, dchi_eV=dchi1 * i))
    r = torch.stack(cols, dim=1)                 # (nd, ni2)
    return r / r.sum(dim=1, keepdim=True)        # normalise so the stages sum to 1''')

md(r"""**Solving for the electron density.** The fixed-point iteration starts from the guess that half the particles are electrons, then repeats three steps until $n_e$ stops moving: recompute the atom density and Debye lowering from the current $n_e$, sum the charge donated by every element, and damp the update by averaging it with the old guess. The constants are in CGS units ($e = 4.801\times10^{-10}\,\mathrm{esu}$, $4\pi = 12.5664$).""")

code(r'''# stage-charge weights [0,1,2,...] as a device tensor, sliced per element below
CHARGE = torch.arange(6, dtype=DTYPE, device=DEVICE)

def solve_electron_density(max_iter=400, tol=1e-10):
    """Iterate the LTE charge-balance equation for n_e over all depths.

    Inputs are the fixed gas pressure, temperature, elemental abundances, partition tables,
    and ionization energies already loaded as tensors. The element loop handles heterogeneous
    atomic data, while each element is evaluated across all atmosphere layers at once.
    Returns the converged electron density in cm^-3.
    """
    n_e = P_gas / tk / 2.0                       # initial guess: half the particles are electrons

    for _ in range(max_iter):
        n_atom = P_gas / tk - n_e                # remaining particles are nuclei (atoms + ions)

        # Debye radius [cm], n_charge ~ 2*n_e for a singly-charged gas; lowering capped at 1 eV
        lam_D = torch.sqrt(tk / (12.5664 * (4.801e-10)**2 * 2.0*n_e))
        dchi1 = torch.clamp(1.44e-7 / lam_D, max=1.0)

        # sum the electron donations over every element
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
jphot = int(torch.argmin(torch.abs(tau - 2/3)))
print(f"electron density at tau=2/3: {float(n_e[jphot]):.3e} cm^-3")
print(f"electron density spans {float(n_e.min()):.3e} -> {float(n_e.max()):.3e} cm^-3")''')

md(r"""The electron density rises by many orders of magnitude inward because the gas pressure rises inward. What matters for the spectrum, however, is not just the absolute density but who supplied the electrons in the line-forming layers.

Now look at where those electrons come from — and how the answer shifts with depth.""")

md(r"""**Attributing the electrons.** Run the same Saha machinery one element at a time, store each element's electron contribution, and collapse it into two competing shares: hydrogen alone, and all metals ($Z \geq 3$). This uses the self-consistent $n_e$ just solved above.""")

code(r'''# electron donation by every element at every depth, evaluated at the solved n_e
n_atom   = P_gas / tk - n_e
n_charge = 2.0 * n_e
dchi1    = torch.clamp(1.44e-7 / torch.sqrt(tk / (12.5664 * (4.801e-10)**2 * n_charge)), max=1.0)

ne_Z = torch.zeros(nd, 100, dtype=DTYPE, device=DEVICE)   # ne_Z[:, Z] = electrons from element Z
for Z in range(1, 100):
    f = ionization_fractions(Z, T, n_e, dchi1)
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

md(r"""The competition turns over with depth. In the **line-forming layers** ($\tau \sim 0.01$--$0.1$, $T \approx 4800\ \mathrm{K}$) the easily-ionized metals supply most of the electrons — Mg, Fe, Si, Na — even though they are a thousand times rarer than hydrogen. This is why metal-poor stars have lower electron pressures and weaker H-minus continuum opacity at the same temperature. Deeper, hydrogen's overwhelming abundance wins; higher up, the very thin gas ionizes hydrogen just enough that even that tiny fraction overtakes the already-saturated metals. The slogan "metals make the electrons" is specifically a statement about the cool line-forming photosphere.""")

md(r"""## Completing the atmosphere

With $n_e$ in hand the atmosphere is ready for opacity work: every depth carries temperature, pressure, density, and electron density. The two-panel portrait below shows the electron density on a log scale and the hydrogen ionized fraction on the same depth grid.""")

code(r'''fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))
xne_np = n_e.detach().cpu().to(torch.float64).numpy()
ax[0].plot(logtau, np.log10(xne_np), color="C0")
ax[0].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[0].set_ylabel(r"$\log_{10} n_e$  [cm$^{-3}$]")
ax[0].set_title("Electron density vs depth")

ax[1].plot(logtau, frac_HII.cpu().numpy(), color="C3")
ax[1].axvline(np.log10(2/3), ls="--", color="0.5", lw=1)
ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$")
ax[1].set_ylabel("ionized fraction of hydrogen"); ax[1].set_title("Hydrogen ionization vs depth")
fig.tight_layout(); plt.show()

n_e_for_opacity = xne_np.copy()
print(f"electron density ready for opacity: {n_e_for_opacity[0]:.3e} (top) -> "
      f"{n_e_for_opacity[-1]:.3e} (bottom) cm^-3")''')

# ── Charge closure ────────────────────────────────────────────────────────
md(r"""## Conservation check

The electron density was not fitted or imported; it was found by demanding that the positive ion charge equal the free-electron density. The simplest internal check is therefore to recompute the electron donation from all elements using the final ionization fractions and ask how well it closes:

$$
\frac{\left|\sum_Z n_Z \sum_i i f_{Z,i} - n_e\right|}{n_e}.
$$

This is a physical conservation check. A small value means the iteration really solved the coupled Saha-plus-charge equation.""")

code(r'''electrons_from_ions = ne_Z[:, 1:].sum(dim=1)
closure = torch.abs(electrons_from_ions - n_e) / torch.clamp(n_e, min=torch.finfo(DTYPE).tiny)
print(f"maximum fractional charge-closure error = {float(closure.max()):.3e}")
print(f"photospheric closure error             = {float(closure[jphot]):.3e}")

if float(closure.max()) > 1e-4:
    raise RuntimeError("charge conservation did not close cleanly; inspect the Saha iteration")''')

md(r"""The closure error is far below any astrophysical uncertainty in this simplified LTE atmosphere. Now the atmosphere carries the electron density needed by opacity calculations, and the result has a direct physical interpretation: in the solar line-forming layers, low-ionization metals supply most of the electrons even though hydrogen dominates the nuclei count.""")

# ════════════════════════════════════════════════════════════════════════════
#  PFSAHA — the full per-ion partition functions and population factors
# ════════════════════════════════════════════════════════════════════════════
md(r"""---

## PFSAHA: per-ion partition functions and population factors

The charge-balance core above gave us $n_e$ with successive stage-to-stage Saha ratios and temperature-tabulated partition functions $U(T)$. Detailed opacity calculations need more than $n_e$: they need the physical stage fraction of every ionization stage, $F_{Z,i}$, and the **partition function of each ion**, $U_{Z,i}$, built from the relevant atomic data. The opacity calculation does not store the raw ion population $n_ZF_{Z,i}$ directly. It stores the population factor

$$
\Phi_{Z,i} = \frac{n_Z F_{Z,i}}{U_{Z,i}}, \qquad n_Z = A_Z\, n_{\rm atom},
$$

because an opacity formula can then multiply this factor by the lower level's statistical weight and Boltzmann factor to get the absorbing population. This is the job of **PFSAHA**: for each depth, element, and ion, assemble the appropriate partition function $U$, run the Saha ladder, and return both $F/U$ and $\Phi$. The name is historical; in these notes it simply means "partition functions plus Saha populations."

The partition function is no longer a single table lookup. Three different machines build it, dispatched by element:

- **Ca--Ni / PFIRON block** ($Z=20$--$28$, including the traditional iron-group elements): a measured $U(\log_{10}T,\ \log_{10}\Delta\chi)$ grid (PFIRON), bilinearly interpolated.
- **Light elements** (H, He, C, O, Na, Mg, Al, Si, Ca, K, B): an explicit sum over measured energy levels $U = \sum_i g_i\,e^{-E_i/kT}$.
- **Everything else**: a packed-integer table (NNN) of bracketed partition values, unpacked and interpolated in temperature.

On top of $U$ sit two corrections. A low-temperature **ground-state floor** prevents a packed table from falling below the ground-state statistical weight. A high-temperature **occupation correction** accounts for the fact that high Rydberg states are dissolved by the plasma environment rather than remaining infinitely sharp bound levels.""")

md(r"""**Smooth formulas and table decisions.** Boltzmann sums, interpolation blends, and Saha ratios are smooth arithmetic. Table lookup is different: choosing a temperature bracket or an occupation-correction regime is a discrete decision. The code therefore separates the physical arithmetic from the table indexing, so a boundary choice does not masquerade as a new piece of physics.""")

md(r"""**Load the PFSAHA atomic-data tables.** The inputs are the depth state PFSAHA consumes — temperature, gas pressure, electron density, atomic nuclei density, and abundances — plus the measured atomic-data tables: packed ordinary-ion partition data, ionization potentials, the iron-group grid, explicit light-element level blocks, and the ground-state floor.""")

code(r'''import math
PF = np.load(pathlib.Path("..") / "reference" / "pfsaha_inputs.npz")

# PFSAHA physical constants in the units used by the atomic-data tables
EV_TO_CM = 8065.479          # 1 eV in cm^-1
LN10     = math.log(10.0)

# depth state and host copies used by table indexing
T_pf = dev(PF["T"])
kT_eV = dev(PF["tkev"])
hc_over_kT = dev(PF["hckt"])
tlog = dev(PF["tlog"])
tk_pf = dev(PF["tk"])
Pg = dev(PF["gas_pressure"])
electron_density_pf = dev(PF["electron_density"])
xnatom = dev(PF["xnatom"])
xab_pf = dev(PF["xabund"])
T_h    = PF["T"].astype(np.float64);    tlog_h = PF["tlog"].astype(np.float64)   # host copies for table indexing
nion_per_Z = PF["nion_per_Z"].astype(np.int64); LOCZ = PF["LOCZ"].astype(np.int64)
ndp = T_h.size

# atomic-data tables: integer-packed tables stay as host arrays; level blocks are tensors
NNN_i      = PF["NNN"].astype(np.int64)            # (6,374) packed partition data (host int)
POTION_h   = PF["POTION"].astype(np.float64)       # (999,) ionization potentials [cm^-1]
SCALE_h    = PF["SCALE"].astype(np.float64)
PFTAB_h    = PF["PFTAB"].astype(np.float64)        # (7,56,10,9) Fe-group grid
PFLO_h     = PF["PFIRON_POTLO"].astype(np.float64)
PFLOLOG_h  = PF["PFIRON_POTLOLOG"].astype(np.float64)
pfg = dev(PF["pfground_tab"])                       # (605,80) ground-state floor, pre-evaluated
SP = {k: dev(v) for k, v in PF.items() if k.startswith("sp_")}   # light-element level blocks
print(f"PFSAHA inputs: {ndp} layers, 99 elements, up to 6 ion stages stored")''')

md(r"""**The Debye lowering, reused.** PFSAHA uses the same continuum-lowering idea as the charge-balance core. The occupation correction switches on only when the per-charge lowering $\Delta\chi \geq 0.1\ \mathrm{eV}$, so the code keeps both the lowering value and the associated table-regime decision explicit.""")

code(r'''def debye_lowering(electron_density, kT_erg, gas_pressure):
    """Per-unit-charge Debye lowering Delta_chi [eV]."""
    charge = 2.0 * electron_density
    excess = 2.0 * electron_density - gas_pressure / kT_erg
    charge = torch.where(excess > 0.0, charge + 2.0 * excess, charge)
    charge = torch.where(charge == 0.0, torch.ones_like(charge), charge)
    debye = torch.sqrt(kT_erg / 2.8965e-18 / charge)
    return torch.clamp(1.44e-7 / debye, max=1.0)

def debye_lowering_np(electron_density, kT_erg, gas_pressure):
    """The same lowering as host arrays for the occupation-regime decision."""
    electron_density = np.asarray(electron_density, np.float64)
    kT_erg = np.asarray(kT_erg, np.float64)
    gas_pressure = np.asarray(gas_pressure, np.float64)
    charge = 2.0 * electron_density
    excess = 2.0 * electron_density - gas_pressure / kT_erg
    charge = np.where(excess > 0.0, charge + 2.0 * excess, charge)
    charge = np.where(charge == 0.0, 1.0, charge)
    debye = np.sqrt(kT_erg / 2.8965e-18 / charge)
    return np.minimum(1.0, 1.44e-7 / debye)

potlow    = debye_lowering(electron_density_pf, tk_pf, Pg)
potlow_h  = debye_lowering_np(PF["electron_density"], PF["tk"], PF["gas_pressure"])
print(f"Debye lowering per charge: {float(potlow.min()):.2e} -> {float(potlow.max()):.3f} eV")''')

md(r"""**The iron-group partition grid (PFIRON).** Ca--Ni read $U$ from a measured grid in $\log_{10}T$ and $\log_{10}\Delta\chi$. This matters because iron-group ions have dense, tangled level structures; a compact grid is more reliable than a short explicit level sum. The helper below brackets temperature and continuum lowering, then bilinearly interpolates the grid for one element and ion stage.""")

code(r'''def pfiron(iz, ion, tlog10_h, potlow_cm1):
    """PFIRON U for one (element, ion), returned over the full depth grid."""
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

md(r"""**The ordinary-ion partition (NNN).** Most elements unpack $U$ from the packed integer table NNN. A temperature bracket selects an encoded pair of endpoint values, and the partition function is interpolated between them. This is atomic-data compression: the physics is still the same partition function $U(T)$.""")

code(r'''def nnn_partition(c0, ip_val):
    """Ordinary-ion U from the packed NNN table. c0 = 0-based column; ip_val = IP [eV]."""
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

md(r"""**The light-element level sums.** H, He, and the common light elements C/O/Na/Mg/Al/Si/Ca/K/B build $U$ as an explicit Boltzmann sum over measured energy levels,

$$
U(T) = U_0 + \sum_i g_i\,e^{-E_i hc/kT},
$$

with a few high-lying terms added where they materially affect the partition function. This is the same formula introduced at the start of the lecture, now applied to curated atomic level lists.""")

code(r'''def _block(Ekey, Gkey, nlev, part0):
    """Boltzmann sum part0 + sum_i g_i exp(-E_i hc_over_kT) over measured levels."""
    E, g = SP["sp_" + Ekey], SP["sp_" + Gkey]
    s = part0.clone() if torch.is_tensor(part0) else torch.full_like(hc_over_kT, float(part0))
    for i in range(1, nlev):
        s = s + g[i] * torch.exp(-E[i] * hc_over_kT)
    return s

def special_partition(col, ion):
    """Per-ion explicit level sum for the special light elements. Returns (PART, g_override, D1)
    or None if `col` is not a special column. g_override is a scalar/None."""
    z = torch.zeros_like(hc_over_kT)
    if col == 2:  return torch.ones_like(hc_over_kT), None, z                          # bare ion (H II-like)
    if col == 1:                                                                 # H I
        return torch.where(T_pf >= 9000.0, _block("EHYD","GHYD",6,2.0), torch.full_like(hc_over_kT,2.0)), None, 109677.576/6.5/6.5*hc_over_kT
    if col == 3:                                                                 # He I
        return torch.where(T_pf >= 15000.0, _block("EHE1","GHE1",29,1.0), torch.ones_like(hc_over_kT)), None, 109677.576/5.5/5.5*hc_over_kT
    if col == 4:                                                                 # He II
        return torch.where(T_pf >= 30000.0, _block("EHE2","GHE2",6,2.0), torch.full_like(hc_over_kT,2.0)), None, 4.0*109722.267/6.5/6.5*hc_over_kT
    if col == 354:                                                               # C I
        p = _block("EC1","GC1",14, 1.0 + 3.0*torch.exp(-16.42*hc_over_kT) + 5.0*torch.exp(-43.42*hc_over_kT))
        p = p + (108.0*torch.exp(-80000.0*hc_over_kT) + 189.0*torch.exp(-84000.0*hc_over_kT) + 247.0*torch.exp(-87000.0*hc_over_kT)
                 + 231.0*torch.exp(-88000.0*hc_over_kT) + 190.0*torch.exp(-89000.0*hc_over_kT) + 300.0*torch.exp(-90000.0*hc_over_kT))
        return p, None, z
    if col == 355:                                                               # C II
        p = _block("EC2","GC2",6, 2.0 + 4.0*torch.exp(-63.42*hc_over_kT))
        p = p + (6.0*torch.exp(-131731.80*hc_over_kT) + 4.0*torch.exp(-142027.1*hc_over_kT) + 10.0*torch.exp(-145550.13*hc_over_kT)
                 + 10.0*torch.exp(-150463.62*hc_over_kT) + 2.0*torch.exp(-157234.07*hc_over_kT) + 6.0*torch.exp(-162500.0*hc_over_kT)
                 + 42.0*torch.exp(-168000.0*hc_over_kT) + 56.0*torch.exp(-178000.0*hc_over_kT) + 102.0*torch.exp(-183000.0*hc_over_kT)
                 + 400.0*torch.exp(-188000.0*hc_over_kT))
        return p, None, z
    if col == 51:                                                                # Mg I
        p = _block("EMG1","GMG1",11, 1.0)
        p = p + (5.0*torch.exp(-53134.0*hc_over_kT) + 15.0*torch.exp(-54192.0*hc_over_kT) + 28.0*torch.exp(-54676.0*hc_over_kT) + 9.0*torch.exp(-57853.0*hc_over_kT))
        return p, 4.0, 109734.83/4.5/4.5*hc_over_kT
    if col == 52:                                                                # Mg II
        p = _block("EMG2","GMG2",6, 2.0)
        p = p + (10.0*torch.exp(-93310.80*hc_over_kT) + 14.0*torch.exp(-93799.70*hc_over_kT) + 6.0*torch.exp(-97464.32*hc_over_kT)
                 + 10.0*torch.exp(-103419.82*hc_over_kT) + 14.0*torch.exp(-103689.89*hc_over_kT) + 18.0*torch.exp(-103705.66*hc_over_kT))
        return p, 2.0, 4.0*109734.83/5.5/5.5*hc_over_kT
    if col == 57:                                                                # Al I
        p = _block("EAL1","GAL1",9, 2.0 + 4.0*torch.exp(-112.061*hc_over_kT))
        p = p + 10.0*torch.exp(-42235.0*hc_over_kT) + 14.0*torch.exp(-43831.0*hc_over_kT)
        return p, 2.0, 109735.08/5.5/5.5*hc_over_kT
    if col == 63:                                                                # Si I
        p = _block("ESI1","GSI1",11, 1.0 + 3.0*torch.exp(-77.115*hc_over_kT) + 5.0*torch.exp(-223.157*hc_over_kT))
        p = p + (76.0*torch.exp(-53000.0*hc_over_kT) + 71.0*torch.exp(-57000.0*hc_over_kT) + 191.0*torch.exp(-60000.0*hc_over_kT)
                 + 240.0*torch.exp(-62000.0*hc_over_kT) + 251.0*torch.exp(-63000.0*hc_over_kT) + 300.0*torch.exp(-65000.0*hc_over_kT))
        return p, None, z
    if col == 64:                                                                # Si II
        p = _block("ESI2","GSI2",6, 2.0 + 4.0*torch.exp(-287.32*hc_over_kT))
        p = p + (6.0*torch.exp(-81231.59*hc_over_kT) + 6.0*torch.exp(-83937.08*hc_over_kT) + 10.0*torch.exp(-101024.09*hc_over_kT)
                 + 14.0*torch.exp(-103556.35*hc_over_kT) + 10.0*torch.exp(-108800.0*hc_over_kT) + 42.0*torch.exp(-115000.0*hc_over_kT)
                 + 6.0*torch.exp(-121000.0*hc_over_kT) + 38.0*torch.exp(-125000.0*hc_over_kT) + 34.0*torch.exp(-132000.0*hc_over_kT))
        return p, 2.0, 4.0*109734.83/4.5/4.5*hc_over_kT
    if col == 96:                                                                # Ca I
        p = _block("ECA1","GCA1",8, 1.0)
        p = p + (28.0*torch.exp(-37000.0*hc_over_kT) + 67.0*torch.exp(-40000.0*hc_over_kT) + 21.0*torch.exp(-43000.0*hc_over_kT) + 34.0*torch.exp(-48000.0*hc_over_kT))
        return p, 4.0, 109734.82/4.5/4.5*hc_over_kT
    if col == 97:                                                                # Ca II
        return _block("ECA2","GCA2",5, 2.0) + 12.0*torch.exp(-68000.0*hc_over_kT), 2.0, 109734.83/4.5/4.5*hc_over_kT
    if col == 367:                                                               # O I
        p = _block("EO1","GO1",13, 5.0 + 3.0*torch.exp(-158.265*hc_over_kT) + torch.exp(-226.977*hc_over_kT))
        p = p + (15.0*torch.exp(-101140.0*hc_over_kT) + 131.0*torch.exp(-103000.0*hc_over_kT) + 128.0*torch.exp(-105000.0*hc_over_kT) + 600.0*torch.exp(-107000.0*hc_over_kT))
        return p, None, z
    if col == 45:                                                                # Na I
        return _block("ENA1","GNA1",8, 2.0) + 10.0*torch.exp(-34548.745*hc_over_kT) + 14.0*torch.exp(-34586.96*hc_over_kT), 2.0, 109734.83/4.5/4.5*hc_over_kT
    if col == 14:                                                                # B I
        p = _block("EB1","GB1",7, 2.0 + 4.0*torch.exp(-15.25*hc_over_kT))
        p = p + (6.0*torch.exp(-57786.80*hc_over_kT) + 10.0*torch.exp(-59989.0*hc_over_kT) + 14.0*torch.exp(-60031.03*hc_over_kT) + 2.0*torch.exp(-63561.0*hc_over_kT))
        return p, 2.0, 109734.83/4.5/4.5*hc_over_kT
    if col == 91:                                                                # K I
        return _block("EK1","GK1",8, 2.0) + 10.0*torch.exp(-27397.077*hc_over_kT) + 14.0*torch.exp(-28127.85*hc_over_kT), 2.0, 109734.83/5.5/5.5*hc_over_kT
    return None''')

md(r"""**Assembling $U$ for one element.** This function ties the three partition-function paths together for a single element $Z$. For each ion stage it reads the ionization potential, builds the raw $U$ (PFIRON, special level sum, or NNN), applies the low-temperature ground-state floor, and adds the high-temperature occupation correction. It returns the stack of $U$, the ionization potentials, and the continuum lowering that the Saha ladder consumes.""")

code(r'''def occupation_term(d, zion, tv):
    """The high-T occupation-probability correction (Hummer-Mihalas style)."""
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
    each (nion2, nd)."""
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
                tv = torch.where(tcap, torch.full_like(T_pf, (T2000*11.0) * KEV), kT_eV)
                d1c = torch.where(D1 <= 0.0, 0.1 / tv, D1); d2c = POTLO[k] / tv
                if G > 0.0:
                    add = G * torch.exp(-ip_val / tv) * (occupation_term(d2c, zion, tv) - occupation_term(d1c, zion, tv))
                    PART[k] = torch.where(gate, PART[k] + add, PART[k])
    return PART, IP, POTLO, nion2''')

md(r"""**The Saha ladder, in log space.** Given $U$, $\chi$, and the lowering for every stage, the Saha ladder chains the stage-to-stage ratios and normalises them. We do this in **log space**: accumulate $\log F$, subtract the per-depth maximum, exponentiate, and normalise. The raw running product of Saha ratios can overflow in the deep, hot layers; clamping would silently break particle conservation, while the log-space softmax keeps the stage fractions normalized. The output is the stored stage factor $F/U$.""")

code(r'''def saha_ladder(PART, IP, POTLO):
    """Return Saha stage factors F/PART using a log-space softmax.

    PART, IP, and POTLO carry one element's partition functions, ionization potentials, and Debye
    lowering for each ion stage. The result is shaped like PART and stores the population factor
    later opacity routines multiply by level weights and Boltzmann terms. The log formulation is
    the stability-critical detail: direct Saha products overflow in the deepest hot layers.
    """
    nion2 = PART.shape[0]
    log_cf = math.log(2.0 * SAHA) + 1.5 * torch.log(T_pf) - torch.log(electron_density_pf)   # (nd,)
    logF = torch.zeros(nion2, ndp, dtype=DTYPE, device=DEVICE)                   # logF[0]=0 (F[0]=1)
    eps = torch.finfo(DTYPE).tiny
    for ion in range(2, nion2 + 1):
        k = ion - 1
        logratio = (log_cf + torch.log(torch.clamp(PART[k], min=eps))
                    - torch.log(torch.clamp(PART[k-1], min=eps))
                    - (IP[k-1] - POTLO[k-1]) / kT_eV)
        logF[k] = torch.where(PART[k-1] > 0.0, logratio, torch.full_like(logratio, -float("inf")))
    logout = torch.cumsum(logF, dim=0)                          # un-normalised log stage population
    w = torch.exp(logout - torch.amax(logout, dim=0, keepdim=True))             # softmax (overflow-free)
    out = w / torch.sum(w, dim=0, keepdim=True)
    return torch.where(PART > 0.0, out / PART, torch.zeros_like(out))           # F/PART''')

md(r"""**Drive PFSAHA over every element.** Each element is one `build_part` plus one `saha_ladder`, stored into the $(\text{depth}, 99, 6)$ result arrays. The six stored stages cover the ion stages that matter for the solar photosphere.""")

code(r'''NSTORE = 6
U_pf    = torch.zeros(ndp, 99, NSTORE, dtype=DTYPE, device=DEVICE)   # per-ion partition functions
popfrac = torch.zeros(ndp, 99, NSTORE, dtype=DTYPE, device=DEVICE)   # per-ion F/U factors
popion  = torch.zeros(ndp, 99, NSTORE, dtype=DTYPE, device=DEVICE)   # opacity population factors n_ion/U

for Z in range(1, 100):
    PART, IP, POTLO, nion2 = build_part(Z, int(nion_per_Z[Z-1]))
    fr = saha_ladder(PART, IP, POTLO)
    ns = min(NSTORE, nion2)
    U_pf[:, Z-1, :ns]    = PART[:ns].T
    popfrac[:, Z-1, :ns] = fr[:ns].T
    popion[:, Z-1, :ns]  = (fr[:ns].T) * (xnatom[:, None] * xab_pf[Z-1])

print(f"PFSAHA assembled: U, F/U, and opacity population factors for "
      f"99 elements x {NSTORE} stages x {ndp} depths")
# the physical stage fraction is F = (F/U)*U; popfrac stores F/U, so re-form F to read off ionization
F_Fe = (popfrac[:, 25, :] * U_pf[:, 25, :])
jp = int(torch.argmin(torch.abs(tau - 2/3)))
print(f"at the photosphere, Fe is {100*float(F_Fe[jp,1]):.1f}% singly ionized "
      f"(Fe II), {100*float(F_Fe[jp,0]):.1f}% neutral (Fe I) — the dominant solar iron stage is Fe II")''')

md(r"""## Reading the PFSAHA state

PFSAHA returns three arrays with related but distinct meanings:

$$
U_{Z,i},\qquad
\frac{F_{Z,i}}{U_{Z,i}},\qquad
\Phi_{Z,i}=\frac{n_ZF_{Z,i}}{U_{Z,i}}.
$$

Multiplying the first two gives the physical ion fraction $F_{Z,i}$. Multiplying $\Phi$ by $U$ gives the actual number density in that ion stage, $n_ZF_{Z,i}$. Summing that number density times the stage charge gives the electron density represented by the **stored stages**. The array stores the first six stages, which are the important ones for the solar photosphere; in the hottest layers, trace population in higher stages can leave a small residual.""")

code(r'''stage_fraction = popfrac * U_pf                         # F_Z,i
stage_density  = popion * U_pf                          # n_Z * F_Z,i
charge_pf = torch.arange(NSTORE, dtype=DTYPE, device=DEVICE)
xne_from_pfsaha = (stage_density * charge_pf[None, None, :]).sum(dim=(1, 2))

electron_closure_pf = torch.abs(xne_from_pfsaha - electron_density_pf) / torch.clamp(electron_density_pf, min=torch.finfo(DTYPE).tiny)
stage_sum = stage_fraction.sum(dim=2)

print(f"PFSAHA electron closure: max fractional error = {float(electron_closure_pf.max()):.3e}")
print(f"PFSAHA electron closure at tau=2/3       = {float(electron_closure_pf[jp]):.3e}")
print(f"stored-stage fraction sum near photosphere: H={float(stage_sum[jp,0]):.6f}, "
      f"Na={float(stage_sum[jp,10]):.6f}, Fe={float(stage_sum[jp,25]):.6f}")

for Z, name in [(1, "H"), (11, "Na"), (12, "Mg"), (26, "Fe")]:
    fr = stage_fraction[jp, Z-1]
    dominant = int(torch.argmax(fr).item())
    print(f"{name:>2s}: dominant stage at tau=2/3 is charge {dominant} "
          f"({100*float(fr[dominant]):.1f}% of that element)")''')

md(r"""These checks translate the arrays back into physical quantities. Near the photosphere, the stored stage fractions sum to one for the relevant elements and the dominant ionization stages are the expected ones: H is neutral, while Na, Mg, and Fe are mostly singly ionized. The maximum electron-closure residual occurs away from the line-forming photosphere, where the stored six-stage representation omits tiny contributions from higher stages. For the opacity calculation, the important result is that we now have not just an electron density, but the per-ion population factors for the stages that absorb.""")


md(r"""## Synthesis: what you built

You closed the atmosphere's particle accounting. Starting from temperature, gas pressure, density, and abundances, you used the **Boltzmann** distribution to populate energy levels, the **Saha** equation to balance ionization stages, **Debye lowering** to account for plasma screening, and **charge conservation** to solve for the electron density at every depth.

Along the way you found the quiet truth of the cool-star photosphere: hydrogen is almost entirely neutral there, and in the line-forming layers the free electrons come from a handful of low-ionization metals. This is not a minor detail. H-minus opacity, the continuum opacity that dominates the optical Sun, is proportional to the supply of free electrons; changing the metal abundance changes the electron pressure and therefore changes the continuum.

You then went one level deeper. Temperature-only partition functions are enough to explain the charge balance, but detailed opacity needs per-ion population factors. PFSAHA assembles each ion's depth-correct $U$ from atomic data, applies the Debye and occupation corrections, and runs the same Saha ladder in log space. The returned arrays say which ion stages exist and how much population each stage contributes to opacity terms.

Those electrons and per-ion factors are the raw material for continuous and line opacity: they decide how much H-minus can form, how many neutral atoms can scatter light, and how many atoms or ions occupy the lower levels of absorbing transitions.""")

md(r"""## Summary

- In LTE the **Boltzmann** distribution sets level populations within an ion,
  $n_i/n_{\rm ion} = (g_i/U)\,e^{-E_i/kT}$, normalised by the **partition function** $U(T)$.
- The **Saha** equation balances ionization stages:
  $n_{i+1}n_e/n_i = 2(U_{i+1}/U_i)(2\pi m_e kT/h^2)^{3/2}e^{-\chi_i/kT}$,
  with the prefactor $2.4148\times10^{15}\,T^{3/2}$.
- The **Debye continuum-lowering** approximation lowers the ionization potential by
  $\Delta\chi \approx 1.44\times10^{-7}/\lambda_D$ per unit charge — small at the surface,
  growing with density, and capped at $1\ \mathrm{eV}$ in this approximation.
- **Charge conservation**,
  $n_e = \sum_Z n_Z \sum_i i\,f_{Z,i}$,
  closes the system and gives the electron density needed by opacity.
- In the solar photosphere hydrogen is only $\sim10^{-4}$ ionized. In the cool
  **line-forming layers metals (Mg, Si, Fe, Na, Ca) supply the electrons**, while hydrogen
  takes over deeper and higher up.
- Temperature-only partition functions are good for the *summed* electron density but not for
  individual opacity population factors. **PFSAHA** assembles each ion's depth-correct $U$ on the fly:
  a Ca--Ni/PFIRON grid that includes the traditional iron-group elements, hand-built level sums for important light elements, packed
  table interpolation for ordinary ions, plus Debye lowering and a high-temperature occupation
  correction.
- The useful PFSAHA outputs are the per-ion partition function $U$, the normalized stage factor
  $F/U$, and the opacity population factor $\Phi=n_ZF/U$.""")

md(r"""## Practice exercises

**1. The Saha turnover for hydrogen.** Hold the electron density fixed at a photospheric value ($\sim10^{13}\ \mathrm{cm^{-3}}$) and compute the ionized fraction of hydrogen from $4000$ to $12000\ \mathrm{K}$. At what temperature does hydrogen become half-ionized, and why is it far below the naive scale $13.6\ \mathrm{eV}/k \approx 158{,}000\ \mathrm{K}$?

**2. Metals as electron donors.** Repeat the electron-donor attribution at a deep layer ($\tau \sim 10$) and at a line-forming layer ($\tau \sim 0.05$). Explain why hydrogen wins deep while metals win where the lines form.

**3. A metal-poor atmosphere.** The electron density depends on metal abundance. Reduce all elements heavier than helium by a factor of ten, then re-solve charge conservation. By how much does the photospheric electron density drop, and what does that imply for H-minus opacity in metal-poor stars?

**4. Temperature-only $U$ versus PFSAHA.** Pick Na I and compare its neutral-stage population factor at the photosphere two ways: first with the simplified temperature-only partition table, and second with the full PFSAHA partition function. Which ingredient — the hand-built level sum, the Debye lowering, the low-temperature ground-state floor, or the occupation correction — most changes this cool, easily-ionized metal?

**5. Discrete tables versus smooth formulas.** In your own words, separate PFSAHA into two categories: smooth arithmetic, such as Boltzmann factors and Saha ratios, and discrete table decisions, such as choosing a partition-function cell. Why is a table-boundary mistake more serious than ordinary round-off? Why is the log-space Saha ladder safer than multiplying raw Saha ratios stage by stage?""")

md(r"""## Further reading

- **Gray, D. F. (2005). *The Observation and Analysis of Stellar Photospheres*, 3rd ed., Cambridge University Press.** Chapter 1 derives the Saha and Boltzmann equations and works the solar electron-donor problem explicitly.
- **Saha, M. N. (1921). *On a Physical Theory of Stellar Spectra*, Proc. R. Soc. Lond. A, 99, 135.** The original ionization equation.
- **Mihalas, D. (1978). *Stellar Atmospheres*, 2nd ed., Freeman.** Chapter 5 on the LTE equation of state, partition functions, and pressure ionization.
- **Hummer, D. G. & Mihalas, D. (1988). *The Equation of State for Stellar Envelopes*, ApJ, 331, 794.** The occupation-probability formalism behind modern pressure-ionization treatments.
- **NIST Atomic Spectra Database.** A modern source for atomic levels, statistical weights, and ionization energies: [https://physics.nist.gov/asd](https://physics.nist.gov/asd).""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT.relative_to(BOOK)} ({len(cells)} cells)")
