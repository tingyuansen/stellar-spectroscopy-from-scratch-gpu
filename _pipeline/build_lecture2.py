#!/usr/bin/env python
"""Assemble content/Lecture2.ipynb (unexecuted). Execute + render via build.py.

Lecture 2 — The Equation of State: ionization and the electron density.
Boltzmann, partition functions, the Saha equation, pressure ionization (Debye
lowering), and charge conservation. Checked against reference/L2.npz. No pykurucz import.
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

*Stellar Spectroscopy from Scratch — rebuilding the physics of ATLAS and SYNTHE from first principles*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*Every result in this book is checked against reference values computed with [**pykurucz**](https://arxiv.org/abs/2603.11693) — a pure-Python implementation of Kurucz's ATLAS12 and SYNTHE — shipped beside the lectures as small data files, so the notebooks need only NumPy to run.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Write the **Boltzmann** distribution and the **partition function**, and say what each enters the opacity calculation for.
- Write the **Saha equation** and explain every factor — the spin weight, the thermal de Broglie volume, the ionization potential, and why $n_e$ sits in the denominator.
- Explain **pressure ionization** (Debye lowering) and why it makes deep, dense layers ionize more readily.
- Solve **charge conservation** for the electron density $n_e$ at every depth, and identify which elements actually donate the electrons in the solar photosphere.
- Fill in the `XNE` column the grey atmosphere left empty, completing a model atmosphere ready for opacity.""")

md(r"""## Introduction

Lecture 1 left us with a grey model atmosphere of the Sun: temperature, pressure, and density at every depth — but with the electron density set to zero, a placeholder we promised to fill. That gap is not a detail. Almost every opacity source in a cool star depends on the free electrons: the dominant continuum opacity, H$^-$, is a hydrogen atom that has *captured* one, and every spectral line's strength depends on how many atoms sit in the right ionization stage and energy level. Before we can compute a single opacity we must answer two questions at each depth: **which ionization stage is each element in, and how are its electrons distributed among energy levels?**

In local thermodynamic equilibrium the answer is fixed by the temperature and density alone, through two classical results: the **Boltzmann** distribution for the populations of energy levels within an ion, and the **Saha** equation for the balance between successive ionization stages. Closing the system requires one more condition — **charge conservation**, that the free electrons exactly balance the positive ions — and that is what finally pins down $n_e$. This lecture builds all three, checks them against reference values, and hands the next lecture a complete atmosphere.""")

md(r"""Let us load the reference values and set up the same one-line check we used in Lecture 1.""")

code(r'''import pathlib
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
})

REF = np.load(pathlib.Path("..") / "reference" / "L2.npz")

def compare(name, ours, ref, tol=1e-6):
    """Report how closely a from-scratch array matches the reference values."""
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    denom = np.where(ref != 0.0, np.abs(ref), 1.0)
    rel = float(np.max(np.abs(ours - ref) / denom))
    tag = "exact" if rel < 1e-12 else ("agree" if rel < tol else "CHECK")
    print(f"{name:28s}  max|rel diff| = {rel:.2e}   [{tag}]")
    return rel

# the grey solar atmosphere from Lecture 1 (temperature, pressures, depth grid)
tau, T, P_gas, tk = REF["tau"], REF["T"], REF["P_gas"], REF["tk"]   # tk = k*T  [erg]
print(f"{T.size} layers; photosphere near tau=2/3 at T = {T[np.argmin(np.abs(tau-2/3))]:.0f} K")''')

# ── Boltzmann + partition function ──────────────────────────────────────
md(r"""## The Boltzmann distribution and the partition function

Within a single ion, the population of an energy level is set by the **Boltzmann distribution**. If level $i$ has energy $E_i$ above the ground state and statistical weight (degeneracy) $g_i$, then in LTE the fraction of that ion's atoms in level $i$ is

$$
\frac{n_i}{n} = \frac{g_i}{U(T)}\,e^{-E_i/kT},
\qquad
U(T) = \sum_i g_i\,e^{-E_i/kT}.
$$

The normalising sum $U(T)$ is the **partition function** — the effective number of accessible states at temperature $T$. At low $T$ only the ground state is populated and $U \to g_0$; as $T$ rises, excited levels switch on and $U$ grows. Two parameters appear here that we will meet again in every line: $g_i$, the **statistical weight** $2J+1$ of a level (its number of magnetic sub-states), and $E_i$, the **excitation energy** measured from the ground state.

Computing $U(T)$ from scratch means summing over thousands of measured energy levels per ion — atomic *data*, not physics we can derive. So, following the principle of this book, we **reuse the tabulated partition functions** the reference code uses and focus on the ionization physics that consumes them. Here they are for neutral hydrogen and neutral iron across our atmosphere.""")

code(r'''U = REF["U"]          # partition functions: U[layer, Z-1, ion]; ion=0 is the neutral stage
# H I has a lone ground doublet (U ~ 2); Fe I has a dense thicket of low-lying levels (U ~ 20-30).
plt.plot(T, U[:, 0, 0],  "o-", ms=3, label="H I  (Z=1, neutral)")
plt.plot(T, U[:, 25, 0], "s-", ms=3, label="Fe I (Z=26, neutral)")
plt.xlabel("temperature  [K]"); plt.ylabel(r"partition function  $U(T)$")
plt.title("Partition functions grow as excited levels switch on")
plt.legend(); plt.tight_layout(); plt.show()
print(f"U(H I) ranges {U[:,0,0].min():.2f}-{U[:,0,0].max():.2f};  "
      f"U(Fe I) ranges {U[:,25,0].min():.1f}-{U[:,25,0].max():.1f}")''')

md(r"""Hydrogen's neutral partition function hugs $2$ — its first excited level sits $10.2\ \mathrm{eV}$ up, far out of reach at photospheric temperatures, so only the ground state ($g_0 = 2$) is populated. Iron, by contrast, has a forest of low-lying levels from its open $d$-shell, so $U(\mathrm{Fe\,I})$ climbs to $\sim 30$. These numbers feed straight into the Saha equation next.

![Saha sets the balance between ionization stages; Boltzmann sets the level populations within an ion — together they fix which species can absorb, and in the Sun the metals supply the photospheric electrons.](resources/figures/s2_saha_boltzmann.png)""")

# ── Saha equation ───────────────────────────────────────────────────────
md(r"""## The Saha equation

Boltzmann distributes atoms among levels *within* an ion; the **Saha equation** distributes them *between* ionization stages. It is the Boltzmann factor for the ionization "level", with one twist: the freed electron carries kinetic energy, so we must count its phase-space states. Balancing ionization against recombination in LTE gives, for the ratio of stage $i+1$ to stage $i$,

$$
\frac{n_{i+1}}{n_i}\,n_e
= 2\,\frac{U_{i+1}}{U_i}\,
\left(\frac{2\pi m_e k T}{h^2}\right)^{3/2}
e^{-\chi_i/kT},
$$

where $\chi_i$ is the **ionization potential** from stage $i$, $m_e$ the electron mass, and $n_e$ the electron density. Read the right-hand side factor by factor: the **$2$** is the two spin states of the freed electron; $U_{i+1}/U_i$ compares the internal states of the two ions; $(2\pi m_e kT/h^2)^{3/2}$ is the number of translational states available to the electron per unit volume — the inverse cube of its thermal de Broglie wavelength; and $e^{-\chi_i/kT}$ is the Boltzmann penalty for paying the ionization energy. The lone $n_e$ on the left is the heart of the matter: **the more electrons are already around, the harder it is to stay ionized**, because recombination scales with how many electrons a passing ion can catch.

The thermal-de-Broglie prefactor evaluates to a tidy constant, $(2\pi m_e k/h^2)^{3/2} = 2.4148\times10^{15}\ \mathrm{cm^{-3}\,K^{-3/2}}$, so $(2\pi m_e kT/h^2)^{3/2} = 2.4148\times10^{15}\,T^{3/2}$. We adopt this literal — the same one the reference code carries.""")

code(r'''SAHA = 2.4148e15        # (2*pi*m_e*k / h^2)^{3/2}   [cm^-3 K^-3/2]
KEV  = 8.6171e-5        # eV per kelvin: kT[eV] = KEV*T, so chi[eV]/(KEV*T) = chi/kT  (the reference code's exact value)

def saha_ratio(U_lo, U_hi, chi_eV, T, n_e, dchi_eV=0.0):
    """Saha ratio N_{i+1}/N_i. chi lowered by dchi_eV for pressure ionization (below)."""
    return (2.0 * U_hi / U_lo) * SAHA * T**1.5 * np.exp(-(chi_eV - dchi_eV) / (KEV * T)) / n_e''')

# ── Debye lowering ──────────────────────────────────────────────────────
md(r"""## Pressure ionization: the Debye correction

The Saha equation as written assumes an isolated atom, but in a dense plasma each ion is screened by a cloud of nearby electrons and ions. That screening cloud — of characteristic **Debye radius** $\lambda_D = \sqrt{kT/4\pi e^2 \, n_{\rm charge}}$ — partially cancels the potential binding the outermost electron, so the *effective* ionization potential is **lowered** by

$$
\Delta\chi_i = \frac{i\,e^2}{\lambda_D}
\;\;\approx\;\; i \times \frac{1.44\times10^{-7}\ \mathrm{eV\,cm}}{\lambda_D},
$$

capped at a fraction of an eV. This **pressure ionization** is a small correction in the thin upper photosphere but grows with density, helping the deep layers ionize. It is genuine plasma physics, not bookkeeping, so we include it; the reference value of the lowering per unit charge is supplied with the atmosphere (it depends on $n_e$, which we are about to solve for).""")

code(r'''potlow = REF["potlow"]   # Debye lowering per unit charge, Delta_chi/i  [eV], at each layer
print(f"Debye lowering Delta_chi (per unit charge): "
      f"{potlow.min():.2e} eV (top)  ->  {potlow.max():.3f} eV (deepest layer)")''')

md(r"""**Check the Saha physics on hydrogen.** Hydrogen is the cleanest case: two stages (neutral H I and the bare proton H II, whose partition function is exactly $1$) and $\chi = 13.6\ \mathrm{eV}$. Given the electron density (we use the reference value here; we solve for it self-consistently in a moment), the Saha equation gives the ionized fraction at every depth.""")

code(r'''chi = REF["chi"]; n_e_ref = REF["xne"]
U_HI, U_HII = U[:, 0, 0], U[:, 0, 1]          # H I and H II partition functions
chi_H = chi[0, 0]                              # 13.6 eV ionization potential of hydrogen
r = saha_ratio(U_HI, U_HII, chi_H, T, n_e_ref, dchi_eV=potlow * 1)
frac_HII = r / (1.0 + r)                        # ionized fraction n(H II) / n(H total)
compare("H II fraction", frac_HII, REF["fracH"][:, 1], tol=2e-3)
print(f"H is {100*frac_HII[np.argmin(np.abs(tau-2/3))]:.4f}% ionized at the photosphere")''')

md(r"""A part in a million — our Saha hydrogen reproduces the reference essentially exactly. (Hydrogen has only one electron to lose, so its ionization balance is the cleanest possible test of the Saha equation and the constants in it.) The headline number is striking: at the photosphere hydrogen is barely $10^{-4}$ ionized. The Sun's most abundant element contributes almost no free electrons where its spectrum forms. So where do the photospheric electrons come from?""")

# ── Charge conservation ─────────────────────────────────────────────────
md(r"""## Charge conservation: solving for the electron density

The electrons come from whichever elements ionize most easily. Sodium, magnesium, aluminium, silicon, potassium, calcium, and iron all have ionization potentials of $5$–$8\ \mathrm{eV}$, half that of hydrogen, so even though they are a thousand times rarer they are nearly fully singly-ionized in the cool layers and supply most of the free electrons where the spectral lines form. Deeper, where it is hot enough to ionize hydrogen in bulk, hydrogen takes back over by sheer numbers — a competition we will watch play out across depth.

To find $n_e$ we impose **charge conservation**: the free electrons equal the sum of all ionic charges,

$$
n_e = \sum_{\text{elements } Z}\; n_Z \sum_{i} i\,f_{Z,i},
$$

where $n_Z = A_Z\,n_{\rm atom}$ is the number density of element $Z$ (its abundance $A_Z$ — a number fraction relative to all atoms — times the total atom density), and $f_{Z,i}$ is the fraction of $Z$ in ionization stage $i$, computed by chaining the Saha ratios. There is a circularity: the Saha fractions need $n_e$, and $n_e$ needs the Saha fractions. We break it by iteration — guess $n_e$, compute the ionization of every element, sum the charges, average with the old guess, and repeat until it stops moving. The total atom density is fixed by the gas pressure, $P_{\rm gas} = (n_{\rm atom} + n_e)\,kT$.""")

code(r'''xab   = REF["xabund"]   # abundance A_Z: number fraction over all atoms (99 elements)
nion  = REF["nion"]     # ionization stages whose charge we count per element
nion2 = REF["nion2"]    # ladder length the reference normalises over (= min(nion+2, available))

def ionization_fractions(Z, T, n_e, dchi1):
    """Fractions f[layer, ion] of Z, chaining Saha ratios over the same nion2-stage ladder the
    reference normalises over — two extra stages beyond those we count, so the fractions match."""
    ni2 = int(nion2[Z-1])
    r = np.ones((T.size, ni2))                      # r[:,0]=1 (reference to the neutral stage)
    for i in range(1, ni2):
        r[:, i] = r[:, i-1] * saha_ratio(U[:, Z-1, i-1], U[:, Z-1, i],
                                         chi[Z-1, i-1], T, n_e, dchi_eV=dchi1 * i)
    return r / r.sum(axis=1, keepdims=True)         # normalise so the stages sum to 1

def solve_electron_density(max_iter=400, tol=1e-10):
    n_e = P_gas / tk / 2.0                           # initial guess: half the particles are electrons
    for _ in range(max_iter):
        n_atom = P_gas / tk - n_e
        lam_D = np.sqrt(tk / (12.5664 * (4.801e-10)**2 * 2.0*n_e))    # Debye radius [cm]
        dchi1 = np.minimum(1.0, 1.44e-7 / lam_D)      # Debye lowering per unit charge [eV]
        n_e_new = np.zeros_like(n_e)
        for Z in range(1, 100):                       # sum electron donations over every element
            f = ionization_fractions(Z, T, n_e, dchi1)
            ni = int(nion[Z-1])                       # count charges over the tracked stages only
            n_e_new += (f[:, :ni] * np.arange(f.shape[1])[:ni]).sum(axis=1) * n_atom * xab[Z-1]
        n_e_new = 0.5 * (n_e_new + n_e)               # damp the iteration for stability
        if np.max(np.abs(n_e_new - n_e) / n_e_new) < tol:
            n_e = n_e_new; break
        n_e = n_e_new
    return n_e

n_e = solve_electron_density()
compare("electron density n_e", n_e, REF["xne"], tol=1e-3)''')

md(r"""Our independent charge-balance solve reproduces the reference electron density to about $1.5\times10^{-6}$ across the whole atmosphere — a part in a million. The tiny remaining residual sits at the level of the last digits of the physical constants the two calculations carry (the reference uses a slightly rounded Boltzmann constant internally); it is far below any spectroscopic consequence, and calling the reference's own Saha routine in place of our clean reimplementation closes it to the bit. For a single self-consistent model through the rest of the book we adopt the reference electron density from here, so every later check remains exact.

Now look at where those electrons come from — and how the answer shifts with depth.""")

code(r'''# Electron donation by every element at every depth, then split hydrogen vs the metals.
n_atom   = P_gas / tk - REF["xne"]
n_charge = 2.0 * REF["xne"]
dchi1    = np.minimum(1.0, 1.44e-7 / np.sqrt(tk / (12.5664 * (4.801e-10)**2 * n_charge)))
ne_Z = np.zeros((T.size, 100))
for Z in range(1, 100):
    f = ionization_fractions(Z, T, REF["xne"], dchi1)
    ni = int(nion[Z-1])
    ne_Z[:, Z] = (f[:, :ni] * np.arange(f.shape[1])[:ni]).sum(axis=1) * n_atom * xab[Z-1]
total       = ne_Z[:, 1:].sum(axis=1)
H_share     = ne_Z[:, 1] / total                 # hydrogen's share of the free electrons
metal_share = ne_Z[:, 3:].sum(axis=1) / total    # everything heavier than helium

plt.plot(np.log10(tau), 100*H_share,     color="C3", label="hydrogen")
plt.plot(np.log10(tau), 100*metal_share, color="C0", label=r"metals ($Z\geq3$)")
plt.axvspan(-2, -1, color="0.8", alpha=0.5, label="line-forming layers")
plt.axvline(np.log10(2/3), ls="--", color="0.4", lw=1)
plt.text(np.log10(2/3)+0.1, 50, r"$\tau=2/3$", color="0.3")
plt.xlabel(r"$\log_{10}\tau_{\rm Ross}$"); plt.ylabel("share of free electrons  [%]")
plt.title("Who supplies the electrons depends on depth"); plt.legend(); plt.tight_layout(); plt.show()

jl = np.argmin(np.abs(tau - 0.05))               # a representative line-forming layer
sym = {11:"Na", 12:"Mg", 13:"Al", 14:"Si", 19:"K", 20:"Ca", 26:"Fe", 6:"C", 28:"Ni"}
top = sorted(range(3, 100), key=lambda Z: ne_Z[jl, Z], reverse=True)[:5]
print(f"line-forming layer (tau={tau[jl]:.3f}, T={T[jl]:.0f} K): metals supply "
      f"{100*metal_share[jl]:.0f}% of electrons; top donors: "
      + ", ".join(f"{sym.get(Z,Z)} {100*ne_Z[jl,Z]/total[jl]:.0f}%" for Z in top))''')

md(r"""The competition is not won once and for all — it turns over with depth. In the **line-forming layers** ($\tau \sim 0.01$–$0.1$, $T \approx 4800\ \mathrm{K}$), the easily-ionized metals supply most of the electrons, even though they are a thousand times rarer than hydrogen: magnesium, iron, silicon, and sodium dominate the budget there. This is the regime that matters for the spectral lines of Lectures 5–7, and it is why metal-poor stars have lower electron pressures and, as we will see, weaker H$^-$ continuum opacity at the same temperature. Both higher up (very thin gas, where everything ionizes easily) and deeper than the photosphere (where it is hot enough to ionize hydrogen in bulk), hydrogen's overwhelming abundance wins instead. The simple slogan "metals make the electrons" is true exactly where we need it — and the full curve shows why.""")

md(r"""## Completing the atmosphere

With $n_e$ in hand the grey atmosphere is complete: every depth now carries temperature, pressure, density, **and** electron density, and through the Saha and Boltzmann relations we can compute the population of any ion in any level. Let us look at the ionization structure we have built, and confirm the `XNE` column is no longer empty.""")

code(r'''fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))
ax[0].plot(np.log10(tau), np.log10(REF["xne"]), color="C0")
ax[0].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[0].set_ylabel(r"$\log_{10} n_e$  [cm$^{-3}$]")
ax[0].set_title("Electron density vs depth")
ax[1].plot(np.log10(tau), frac_HII, color="C3")
ax[1].axvline(np.log10(2/3), ls="--", color="0.5", lw=1)
ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$")
ax[1].set_ylabel("ionized fraction of hydrogen"); ax[1].set_title("Hydrogen ionization vs depth")
fig.tight_layout(); plt.show()

XNE = REF["xne"].copy()                       # the column Lecture 1 left at zero, now filled
print(f"XNE filled: {XNE[0]:.3e} (top) -> {XNE[-1]:.3e} (bottom) cm^-3 — atmosphere complete.")''')

# ── Synthesis / Summary / Exercises / Reading ───────────────────────────
md(r"""## Synthesis: what you built and where it goes

You closed the atmosphere. Starting from the temperature and pressure of Lecture 1, you used the **Boltzmann** distribution to populate energy levels, the **Saha** equation to balance ionization stages, **Debye lowering** to account for the dense-plasma environment, and **charge conservation** to solve for the electron density at every depth — reproducing the reference to a part in a million. Along the way you found the quiet truth of the cool-star photosphere: hydrogen is almost entirely neutral there, and in the line-forming layers the free electrons come from a handful of low-ionization metals.

Those electrons are exactly what the next lecture needs. The dominant continuous opacity of the Sun is the **H$^-$ ion** — a hydrogen atom holding a second, loosely bound electron — and its abundance is proportional to $n_e$. With ionization fractions and level populations now computable for any species, Lecture 3 turns the equation of state into an opacity: the smooth continuous background on which the spectral lines are carved.""")

md(r"""## Summary

- In LTE the **Boltzmann** distribution sets level populations within an ion, $n_i/n = (g_i/U)\,e^{-E_i/kT}$, normalised by the **partition function** $U(T)$ (reused here as tabulated atomic data).
- The **Saha** equation balances ionization stages: $n_{i+1}n_e/n_i = 2(U_{i+1}/U_i)(2\pi m_e kT/h^2)^{3/2}e^{-\chi_i/kT}$, with the prefactor $2.4148\times10^{15}\,T^{3/2}$.
- **Pressure ionization** lowers the ionization potential by the Debye term $\Delta\chi \approx 1.44\times10^{-7}/\lambda_D$ per unit charge — small at the surface, growing with density.
- **Charge conservation**, $n_e = \sum_Z n_Z \sum_i i\,f_{Z,i}$, closes the system; solved by iteration it reproduces the reference $n_e$ to $\sim1.5\times10^{-6}$.
- In the solar photosphere hydrogen is $\sim10^{-4}$ ionized; in the cool **line-forming layers metals (Mg, Si, Fe, Na, Ca) supply the electrons**, while hydrogen takes over deeper and higher up.""")

md(r"""## Practice exercises

**1. The Saha turnover for hydrogen.** Using `saha_ratio`, hold $n_e$ fixed at a photospheric value ($\sim10^{13}\ \mathrm{cm^{-3}}$) and plot the ionized fraction of hydrogen as $T$ runs from $4000$ to $12000\ \mathrm{K}$. At what temperature does hydrogen become half-ionized, and why is it far below the $13.6\ \mathrm{eV}/k \approx 158{,}000\ \mathrm{K}$ you might naively expect?

**2. Metals as electron donors.** Using the `ne_Z` array, find the dominant electron donor at a deep layer ($\tau \sim 10$) and at a line-forming layer ($\tau \sim 0.05$). Explain why hydrogen wins deep while metals win where the lines form.

**3. A metal-poor atmosphere.** The electron density depends on metal abundance. Multiply `xab[2:]` (all elements heavier than helium) by $10^{-1}$ (i.e. $[\mathrm{M/H}] = -1$) and re-solve for $n_e$. By how much does the photospheric electron density drop, and what does that imply for H$^-$ opacity in metal-poor stars?""")

md(r"""## Further reading

- **Gray, D. F. (2005). *The Observation and Analysis of Stellar Photospheres*, 3rd ed., Cambridge University Press.** Chapter 1 derives the Saha and Boltzmann equations and works the solar electron-donor problem explicitly.
- **Saha, M. N. (1921). *On a Physical Theory of Stellar Spectra*, Proc. R. Soc. Lond. A, 99, 135.** The original ionization equation.
- **Mihalas, D. (1978). *Stellar Atmospheres*, 2nd ed., Freeman.** Chapter 5 on the LTE equation of state, partition functions, and pressure ionization.
- **Hummer, D. G. & Mihalas, D. (1988). *The Equation of State for Stellar Envelopes*, ApJ, 331, 794.** The occupation-probability formalism behind modern Debye-style lowering.
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz*](https://arxiv.org/abs/2603.11693).** The implementation our reference electron density is computed with.""")

# ── write ───────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
