#!/usr/bin/env python
"""Assemble content/Lecture1.ipynb (unexecuted). Execute + render via build.py.

Lecture 1 — Overview & a First Model Atmosphere:
pipeline overview, units/constants, the Planck function, optical depth & LTE, and
a grey model atmosphere built from (Teff, logg). Each result is checked against
reference values saved in reference/L1.npz (precomputed once with the reference
code; see _pipeline/make_references.py). The notebook imports no pykurucz.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture1.ipynb"

cells = []
def md(src): cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

# ── Title + front matter + objectives (one cell, so the callout lifts) ───
md(r"""# Lecture 1 — Overview & a First Model Atmosphere

*Stellar Spectroscopy from Scratch — rebuilding the physics of ATLAS and SYNTHE from first principles*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*Every result in this book is checked, bit for bit, against reference values computed with [**pykurucz**](https://arxiv.org/abs/2603.11693) — a pure-Python implementation of Kurucz's ATLAS12 and SYNTHE — and shipped beside the lectures as small data files, so the notebooks need only NumPy to run.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- State what a synthetic stellar spectrum is, and name the two stages — the **model atmosphere** and the **spectral synthesis** — that turn a star's parameters into a spectrum.
- Write the **Planck function** in the form a real synthesis code uses, and check it against reference values to machine precision.
- Define the **optical depth** $\tau$ and explain why the visible photosphere sits near $\tau \approx 2/3$.
- Build a **grey model atmosphere** — the run of temperature, pressure, and density with depth — from nothing but $T_{\rm eff}$ and $\log g$, and check it against reference values layer by layer.
- Reimplement a piece of physics in plain NumPy and verify it against precomputed reference data — the working rhythm of every lecture that follows.""")

# ── Introduction ────────────────────────────────────────────────────────
md(r"""## Introduction

A stellar spectrum — the star's brightness as a function of wavelength — is the richest measurement we can make of a star we will never visit. Encoded in the depths and shapes of its absorption lines are the photosphere's temperature, its surface gravity, and the abundance of every element that leaves a fingerprint in the light. Reading those numbers off an observed spectrum is an *inverse* problem, and the only way to solve it reliably is to be able to solve the *forward* problem first: given a star's parameters, compute the spectrum it should produce.

That forward calculation is what this course builds, from the ground up. Its lineage is **Kurucz's ATLAS and SYNTHE** — the model-atmosphere and spectral-synthesis codes that have underpinned quantitative stellar spectroscopy for four decades. Those codes are correct and fast, but they are also tens of thousands of lines of hardened, decades-old machinery. Our aim is to recover the *physics* inside them in code short enough to read in a sitting, while losing none of the *accuracy* — and to prove we have lost none by checking every step, to the bit, against reference values precomputed with a faithful implementation of those codes.

The forward problem splits cleanly into two stages. First, a **model atmosphere**: the run of temperature, pressure, and density with depth, fixed by the requirement that the atmosphere be in hydrostatic and radiative equilibrium. Second, **spectral synthesis**: given that structure, compute how much light of each wavelength escapes the surface, by adding up the opacity of every spectral line and solving the radiative-transfer equation. We treat the atmosphere as given for the first half of the course, build the full spectrum, and then return in Part IV and construct the atmosphere itself.""")

md(r"""## The plan: from parameters to photons

The whole pipeline is a chain that turns a handful of numbers — effective temperature $T_{\rm eff}$, surface gravity $\log g$, and a chemical composition — into a flux spectrum $F_\lambda$:

$$
(T_{\rm eff},\ \log g,\ \text{abundances})
\;\longrightarrow\;
\underbrace{T(\tau),\,P(\tau),\,\rho(\tau)}_{\text{model atmosphere}}
\;\longrightarrow\;
\underbrace{n_{\rm ion},\,n_e}_{\text{equation of state}}
\;\longrightarrow\;
\underbrace{\kappa_\lambda}_{\text{opacity}}
\;\longrightarrow\;
\underbrace{F_\lambda}_{\text{radiative transfer}}
$$

Each arrow is one or two lectures:

- **Lecture 1 (here):** the foundations — units, the Planck function, optical depth — and a first model atmosphere $T(\tau), P(\tau)$.
- **Lecture 2:** the equation of state — Saha and Boltzmann give the ionization and the electron density $n_e$.
- **Lecture 3:** the continuous opacity — H$^-$, hydrogen, scattering — the smooth background, from the physics to the production tables that reproduce it to the bit.
- **Lectures 4–5:** line opacity — one spectral line, then the million-line list.
- **Lecture 6:** hydrogen lines — the linear Stark broadening that needs its own engine.
- **Lectures 7–8:** radiative transfer — solve for the emergent flux, then the production JOSH engine, and reproduce a real spectrum.
- **Part IV (9–10):** build the atmosphere itself, closing the loop back to the first arrow.
- **Part V (11–12):** molecules, and the complications real stars add.

We target the **Sun** over a narrow window, $500$–$510\ \mathrm{nm}$, where the spectrum is a forest of atomic absorption lines and the physics is at its cleanest. Later we widen the window and cool the star until molecules take over. Throughout, we reimplement each piece in plain NumPy, reuse the same physical constants and data tables, and **check the result against reference values** — temperatures, pressures, opacities, and ultimately a full spectrum — precomputed once and saved beside the book. So we know the physics is right, not merely plausible.

![The forward pipeline this course rebuilds: four stellar parameters become a spectrum, one stage per lecture, each benchmarked against the reference code.](resources/figures/s1_pipeline.png)""")

md(r"""Let us set up, and load the reference values for this lecture. They live in a small data file next to the notebook; a one-line helper reports how closely a from-scratch array matches them.""")

code(r'''import pathlib
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
})

REF = np.load(pathlib.Path("..") / "reference" / "L1.npz")   # reference values for this lecture

def compare(name, ours, ref, tol=1e-6):
    """Report how closely a from-scratch array matches the reference values."""
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    denom = np.where(ref != 0.0, np.abs(ref), 1.0)           # avoid divide-by-zero on exact zeros
    rel = float(np.max(np.abs(ours - ref) / denom))
    tag = "exact" if rel < 1e-12 else ("agree" if rel < tol else "CHECK")
    print(f"{name:28s}  max|rel diff| = {rel:.2e}   [{tag}]")
    return rel

print("reference values loaded:", ", ".join(REF.files))''')

# ── Units & constants ───────────────────────────────────────────────────
md(r"""## Units and constants

Stellar-atmosphere physics is done in **Gaussian CGS** units — centimetres, grams, seconds — and so will we, because matching the reference to the last bit means matching its unit system and even its numerical *values* of the physical constants. Three constants carry most of the radiation physics:

| symbol | name | value (CGS) | role |
|---|---|---|---|
| $h$ | Planck constant | $6.62607015\times10^{-27}\ \mathrm{erg\,s}$ | sets the energy of a photon, $E=h\nu$ |
| $c$ | speed of light | $2.99792458\times10^{10}\ \mathrm{cm\,s^{-1}}$ | relates wavelength and frequency, $\lambda\nu=c$ |
| $k$ | Boltzmann constant | $1.380649\times10^{-16}\ \mathrm{erg\,K^{-1}}$ | sets the thermal energy scale, $kT$ |

These are the 2019-SI exact values in CGS — the same literals the reference calculation uses — so that no discrepancy can ever be blamed on a constant.""")

code(r'''H = 6.62607015e-27   # Planck constant   [erg s]
C = 2.99792458e10    # speed of light    [cm s^-1]
K = 1.380649e-16     # Boltzmann constant[erg K^-1]

# A photon's energy at 500 nm, and the thermal energy at the solar surface, for scale:
lam = 500e-7                       # 500 nm expressed in cm
print(f"E(500 nm) = h*c/lam = {H*C/lam:.3e} erg = {H*C/lam/1.602e-12:.3f} eV")
print(f"kT at 5770 K        = {K*5770:.3e} erg = {K*5770/1.602e-12:.3f} eV")
print(f"Planck prefactor 2h/c^2 = {2*H/C**2:.6e}  (we will meet this again below)")''')

# ── Planck ──────────────────────────────────────────────────────────────
md(r"""## Blackbody radiation and the Planck function

A cavity in thermal equilibrium at temperature $T$ radiates with a spectrum that depends on $T$ alone — the **Planck function**. Per unit frequency it is the specific intensity

$$
B_\nu(T) = \frac{2h\nu^3}{c^2}\,\frac{1}{e^{h\nu/kT}-1}
\qquad [\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}],
$$

and per unit wavelength, using $B_\lambda\,|d\lambda| = B_\nu\,|d\nu|$ with $\nu = c/\lambda$,

$$
B_\lambda(T) = \frac{2hc^2}{\lambda^5}\,\frac{1}{e^{hc/\lambda kT}-1}.
$$

The Planck function matters here for two reasons. First, it is the **source function** of an atmosphere in local thermodynamic equilibrium: each parcel of gas emits as a blackbody at its own temperature, $S_\lambda = B_\lambda(T)$ — the fact that turns a temperature structure into a spectrum. Second, it fixes the scale: the emergent flux of a star is, to first approximation, $B_\lambda(T_{\rm eff})$ shaped by opacity.

Two limits are worth holding onto. When $h\nu \ll kT$ (radio, far-IR) the exponential linearises and $B_\nu \to 2\nu^2 kT/c^2$ — the **Rayleigh–Jeans** law, independent of $h$. When $h\nu \gg kT$ (the blue/UV side of an optical spectrum) the $-1$ is negligible and $B_\nu \to (2h\nu^3/c^2)\,e^{-h\nu/kT}$ — the **Wien** tail, which falls off exponentially and makes the blue side of a line forest exquisitely temperature-sensitive.""")

md(r"""Synthesis codes do not evaluate $B_\nu$ from the textbook expression; they use an algebraically identical but numerically cleaner form. With $x \equiv h\nu/kT$ and the frequency measured in units of $10^{15}\ \mathrm{Hz}$ ($\nu_{15} \equiv \nu/10^{15}$), Kurucz writes

$$
B_\nu = (1.47439\times10^{-2})\;\nu_{15}^{3}\;\frac{e^{-x}}{1 - e^{-x}}.
$$

Two things are going on. Dividing through by $e^{x}$ replaces $1/(e^{x}-1)$ with $e^{-x}/(1-e^{-x})$, so the exponential is always $\le 1$ and never overflows on the Wien tail. And the constant $1.47439\times10^{-2}$ is $2h/c^2$ with $\nu$ rescaled by $10^{15}$, supplying the missing $10^{45}$ in $(\nu/10^{15})^3$ — to the digits Kurucz tabulated: the cell above prints the exact 2019-SI value, $2h/c^2 = 1.4745\times10^{-47}$, whereas the codes have carried the slightly rounded $1.47439$ for decades. We keep that literal, not the exact value, so our Planck function matches the reference to the last bit — reproducing an established calculation means adopting its constants, historical roundings and all. The factor $e^{-x}/(1-e^{-x})$ is the **photon occupation number** $1/(e^{x}-1)$ rewritten — the mean number of photons per field mode in thermal equilibrium. Its denominator $1-e^{-x}$ is the **stimulated-emission factor**, which reappears on its own when we correct line opacities for stimulated emission in Lecture 4. We implement exactly this form.""")

code(r'''def planck_nu(freq_hz, temperature):
    """Planck B_nu(T) in CGS [erg s^-1 cm^-2 Hz^-1 sr^-1], Kurucz's overflow-safe form.

    freq_hz : photon frequency nu [Hz]      temperature : array of T [K]
    """
    x = H * freq_hz / (K * np.asarray(temperature, float))   # x = h*nu / (k*T), dimensionless
    ehvkt = np.exp(-x)                                        # e^{-x} <= 1, never overflows
    return 1.47439e-2 * (freq_hz / 1e15)**3 * ehvkt / (1.0 - ehvkt)   # photon occupation factor''')

md(r"""Let us look at it: $B_\lambda(T)$ for three temperatures bracketing the Sun. As $T$ rises the curve lifts at every wavelength and its peak slides blueward (Wien's displacement law, $\lambda_{\rm peak}T = \text{const}$). Our $500$–$510\ \mathrm{nm}$ window sits just blueward of the solar peak.""")

code(r'''lam_nm = np.linspace(200, 2000, 400)
lam_cm = lam_nm * 1e-7
for T in (4500, 5770, 7500):
    Bnu = planck_nu(C / lam_cm, T)        # B_nu at each wavelength
    Blam = Bnu * C / lam_cm**2            # convert to per-wavelength: B_lambda = B_nu * c/lambda^2
    plt.plot(lam_nm, Blam, label=f"T = {T} K")
plt.axvspan(500, 510, color="0.5", alpha=0.25, label="our window")
plt.xlabel("wavelength  [nm]"); plt.ylabel(r"$B_\lambda(T)$  [erg s$^{-1}$ cm$^{-2}$ cm$^{-1}$ sr$^{-1}$]")
plt.title("The Planck function across the optical"); plt.legend(); plt.tight_layout(); plt.show()''')

md(r"""Now the first comparison. We evaluate the Planck function at mid-window across a spread of temperatures spanning a stellar atmosphere, and check it against the reference values loaded earlier. Because we use the identical expression and constants, this should match to the bit — a reassurance that our comparison method works before we reach physics where the agreement is not guaranteed.""")

code(r'''freq = float(REF["planck_freq"])                 # 505 nm expressed as a frequency [Hz]
compare("Planck B_nu(505 nm)", planck_nu(freq, REF["planck_T"]), REF["planck_B"])''')

md(r"""Exact agreement — we reproduced the reference Planck function bit for bit. That is the pattern for the whole course in miniature: state the physics, write the clean code, check it against the reference.""")

# ── LTE ─────────────────────────────────────────────────────────────────
md(r"""## Local thermodynamic equilibrium

Everything above assumed the gas radiates like a blackbody at its local temperature. That assumption is **local thermodynamic equilibrium (LTE)**: at each depth the matter is described by a single temperature $T$, so the ionization balance follows the Saha equation (Lecture 2), the level populations follow the Boltzmann distribution, and the source function is $S_\lambda = B_\lambda(T)$. LTE does *not* assume the radiation field itself is a blackbody — photons stream and leak, and computing that leakage is the radiative-transfer problem of Lecture 7 — only that the *matter* is in equilibrium with the local temperature.

LTE holds where collisions are frequent enough to keep populations thermal, which is true through most of a cool star's photosphere — deep enough that the gas is dense, shallow enough that we still see it. It weakens high in the atmosphere and in strong resonance lines, where the departures are called **non-LTE**; we name where it breaks in the final lecture. For the Sun in our window LTE is an excellent approximation, and the codes we follow — like ATLAS and SYNTHE — are LTE codes. We adopt LTE throughout.""")

# ── Optical depth ───────────────────────────────────────────────────────
md(r"""## Optical depth and the photosphere

How deep into a star can we see? The natural depth coordinate is not geometric height $z$ but **optical depth** $\tau$, which counts how many times a photon is, on average, absorbed or scattered. Along a ray, with opacity $\kappa_\lambda$ (cross-section per gram) and mass density $\rho$,

$$
d\tau_\lambda = -\kappa_\lambda\,\rho\,dz ,
$$

the minus sign making $\tau$ increase *inward* from $\tau=0$ at the surface. Because opacity is per gram, it is often cleanest to use the **column mass** $\rho x$ (grams of material above a square centimetre) as the depth variable; the model atmospheres we work with are tabulated against exactly this quantity, called `RHOX`, and then $d\tau_\lambda = \kappa_\lambda\,d(\rho x)$.

A photon escapes from roughly where $\tau \approx 1$ — deeper than that, it is reabsorbed before reaching the surface; shallower, there is too little material above to matter. The precise result, from solving the transfer equation for a grey atmosphere (below), is that the emergent flux forms near

$$
\tau \approx 2/3,
$$

the **Eddington–Barbier** depth. This is why "the photosphere" is a layer, not a surface: at each wavelength we see down to where $\tau_\lambda \approx 2/3$, and because $\kappa_\lambda$ spikes inside a spectral line, lines let us see only the cool, high layers — which is why they appear in absorption. Holding onto "$\tau=2/3$ is where we see" will carry you through the rest of the course.

![Optical depth: at each wavelength we see down to where $\tau \approx 2/3$; photons born deeper are reabsorbed before reaching the surface.](resources/figures/s1_optical_depth.png)""")

# ── Grey atmosphere ─────────────────────────────────────────────────────
md(r"""## A first model atmosphere: the grey atmosphere

To compute a spectrum we need the atmosphere's structure: temperature, pressure, and density as functions of depth. Where does that structure come from? The full answer — solving for radiative and hydrostatic equilibrium with the real, wavelength-dependent opacity — is Part IV of this course. But there is a classic closed-form starting point that needs none of that machinery: the **grey atmosphere**, in which the opacity is assumed independent of wavelength ($\kappa_\lambda \to \kappa$). It is crude, but it is self-contained, it requires only $T_{\rm eff}$ and $\log g$, and — as we will see — it is exactly the model ATLAS12 starts its own iteration from. It is a natural first atmosphere to build the rest of the course on.

For a grey atmosphere in radiative equilibrium, the transfer equation can be solved for the temperature structure exactly. The result is

$$
T^4(\tau) = \tfrac{3}{4}\,T_{\rm eff}^4\,\big[\tau + q(\tau)\big],
$$

where $q(\tau)$ is the **Hopf function**, slowly rising from $q(0)=1/\sqrt{3}\approx0.577$ to $q(\infty)\approx0.710$. Replacing it with the constant $q=2/3$ is the textbook **Eddington approximation**, $T^4=\tfrac34 T_{\rm eff}^4(\tau+\tfrac23)$. Under that approximation the boundary value is *exact*: at $\tau=2/3$, $T^4=\tfrac34 T_{\rm eff}^4(\tfrac23+\tfrac23)=T_{\rm eff}^4$, so $T(2/3)=T_{\rm eff}$ — the effective temperature is, by construction, the temperature at the photosphere. (With Kurucz's exact-Hopf fit instead of the constant $2/3$, $q(2/3)\approx0.70$ and $T(2/3)$ lands a fraction of a percent above $T_{\rm eff}$, as the next cell shows.)

Kurucz uses a compact polynomial fit to the Hopf function,

$$
T(\tau) = T_{\rm eff}\,\Big[\tfrac{3}{4}\big(0.710 + \tau - 0.1331\,e^{-3.4488\,\tau}\big)\Big]^{1/4},
$$

and we adopt the identical fit so we can match the reference layer for layer.""")

code(r'''def grey_temperature(teff, tau):
    """Eddington-Kurucz grey T(tau): a polynomial fit to the Hopf function q(tau)."""
    q = 0.710 + tau - 0.1331 * np.exp(-3.4488 * tau)   # Kurucz's Hopf-function fit
    return teff * (0.75 * q)**0.25

TEFF, LOGG = 5770.0, 4.44          # the Sun: effective temperature [K], log surface gravity [cgs]
g_cgs = 10.0**LOGG                 # surface gravity g [cm s^-2]

# The atmosphere is tabulated on an 80-layer grid, equally spaced in log(tau_Rosseland):
# tau runs from 10^-6.875 (top) to 10^3 (bottom) in steps of 0.125 dex -- the ATLAS12 default.
j = np.arange(80)
tau = 10.0**(-6.875 + 0.125 * j)
T = grey_temperature(TEFF, tau)
print(f"layers: {T.size}   tau: {tau[0]:.2e} .. {tau[-1]:.2e}")
print(f"T(top) = {T[0]:.1f} K    T(tau=2/3) ~ {grey_temperature(TEFF, 2/3):.1f} K    T(bottom) = {T[-1]:.1f} K")''')

md(r"""The grid spans nearly ten decades of optical depth, from $\tau\sim10^{-7}$ at the top to $\tau=10^{3}$ deep below the photosphere, sampled at $0.125$ dex over 80 layers — the literal ATLAS12 card defaults, fine enough to integrate the transfer equation accurately. The printed $T(\tau=2/3)\approx5802$ K sits $0.6\%$ above $T_{\rm eff}=5770$ K; the small excess is the Kurucz fit's $q(2/3)\approx0.70$ rather than the Eddington $2/3$, and the photosphere still sits at $T\approx T_{\rm eff}$ as the grey solution requires. Plotting $T$ against $\log\tau$ shows the temperature rising monotonically inward, steeply near the photosphere and flattening into the optically thin layers above.""")

code(r'''plt.plot(np.log10(tau), T, color="C3")
plt.axvline(np.log10(2/3), ls="--", color="0.4", lw=1)
plt.text(np.log10(2/3)+0.1, T.min()+300, r"photosphere, $\tau=2/3$", color="0.3")
plt.xlabel(r"$\log_{10}\tau_{\rm Ross}$"); plt.ylabel("temperature  [K]")
plt.title(r"Grey temperature structure of the Sun ($T_{\rm eff}=5770$ K)")
plt.tight_layout(); plt.show()''')

md(r"""**First atmosphere check.** The reference grey temperature was built with the identical Hopf-function fit on the same optical-depth grid, so both the grid and the temperatures should match exactly.""")

code(r'''compare("grey tau grid", tau, REF["grey_tau"])
compare("grey T(tau)",   T,   REF["grey_T"])''')

md(r"""## Hydrostatic equilibrium: pressure and density

Temperature is half the structure; we also need pressure and density. These come from **hydrostatic equilibrium** — the photosphere neither collapses nor expands, so at every depth the pressure gradient supports the overlying weight. Writing the balance against optical depth, with $g$ the surface gravity,

$$
\frac{dP_{\rm total}}{d\tau} = \frac{g}{\kappa} .
$$

The total pressure has a gas part and a radiation part, $P_{\rm total} = P_{\rm gas} + P_{\rm rad}$, with $P_{\rm rad} = \tfrac{4\sigma}{3c}T^4$ (tiny in a cool dwarf, but the codes carry it). On the **grey cold start** there is not yet an opacity table, so $\kappa$ is set to a constant placeholder of unity — a deliberately crude bootstrap that ATLAS then refines over its iterations, and that we rebuild properly, with the real Rosseland-mean opacity, in Lecture 10. With $\kappa\equiv1$ the equation integrates trivially: $P_{\rm total} = g\,\tau$, and since the column mass is $\rho x = P_{\rm total}/g$, we get the clean identity $\rho x = \tau$. The electron density is left at zero for now — it is the first thing the equation of state computes in Lecture 2.""")

code(r'''# Hydrostatic equilibrium on the grey cold start: kappa_Ross == 1  =>  P_total = g * tau
P_total = g_cgs * tau                      # total pressure [dyn cm^-2]
RHOX    = P_total / g_cgs                   # column mass [g cm^-2]  (numerically equal to tau here)

P_rad = 2.521e-15 * np.maximum(T**4, TEFF**4 / 2.0)   # radiation pressure, Kurucz's floor form
P_rad = P_rad - P_rad[0]                               # measured relative to the top boundary
P_gas = P_total - P_rad                               # gas pressure -> feeds the EOS in Lecture 2

XNE = np.zeros_like(tau)                    # electron density [cm^-3]: 0 until the EOS fills it (L2)

print(f"{'log tau':>8} {'T [K]':>9} {'P_gas [dyn/cm2]':>16} {'RHOX [g/cm2]':>13}")
for i in (0, 20, 40, 60, 79):
    print(f"{np.log10(tau[i]):8.3f} {T[i]:9.1f} {P_gas[i]:16.4e} {RHOX[i]:13.4e}")''')

md(r"""These columns — `RHOX`, `T`, `P_gas`, `XNE` (the electron density, in $\mathrm{cm^{-3}}$), and the opacity `ABROSS` — are the core of the layer structure a Kurucz `.atm` file stores, and the input the spectral synthesis will consume. (The full deck carries a few more columns — radiative acceleration, turbulent and convective velocities — that we do not need yet.) Reading them top to bottom: at the surface the gas is thin (a few times $10^{-3}\ \mathrm{dyn\,cm^{-2}}$) and cool; ten decades deeper the pressure has climbed to $10^{7}\ \mathrm{dyn\,cm^{-2}}$ and the temperature past $10^4$ K. This is the atmosphere we will pour opacity into.

**Full-structure check.** We compare our gas pressure and column mass against the reference grey structure.""")

code(r'''compare("P_gas", P_gas, REF["grey_pgas"], tol=1e-4)   # tol relaxed: see the note below
compare("RHOX",  RHOX,  REF["grey_rhox"], tol=1e-4)''')

md(r"""The temperature matched to the bit; the pressure and density agree to $\sim2\times10^{-5}$. That residual is not physics — it is the difference between our one-line analytic integral $P_{\rm total}=g\tau$ and the multi-step predictor–corrector the reference uses to integrate the same hydrostatic equation in log-pressure. When we rebuild hydrostatic equilibrium in full in Lecture 9, reproducing the exact predictor–corrector integrator, the last digits fall into place. For now we have a complete, self-consistent grey model atmosphere of the Sun, built from two numbers, agreeing with the reference to one part in $10^{5}$.""")

code(r'''fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))
ax[0].plot(np.log10(tau), np.log10(P_gas), color="C0")
ax[0].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[0].set_ylabel(r"$\log_{10}\,P_{\rm gas}$  [dyn cm$^{-2}$]")
ax[0].set_title("Gas pressure")
ax[1].plot(np.log10(tau), np.log10(RHOX), color="C2")
ax[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[1].set_ylabel(r"$\log_{10}\,\rho x$  [g cm$^{-2}$]")
ax[1].set_title("Column mass")
fig.suptitle("Grey solar atmosphere: the structure we carry into Lecture 2")
fig.tight_layout(); plt.show()''')

# ── Synthesis / Summary / Exercises / Reading ───────────────────────────
md(r"""## Synthesis: what you built and where it goes

You built the scaffolding for everything that follows. You fixed the unit system and the physical constants, and you reproduced the **Planck function** to the bit — the source function that will turn a temperature into a spectrum. You learned that depth is measured in **optical depth**, that we see to $\tau\approx2/3$, and that **LTE** lets the local temperature set the state of the gas. And you built a **grey model atmosphere** of the Sun — $T(\tau)$, $P_{\rm gas}(\tau)$, $\rho x(\tau)$ — from $T_{\rm eff}$ and $\log g$ alone, matching the reference layer by layer.

That atmosphere has one conspicuous gap: the electron density is still zero. Filling it — working out, at each depth, which atoms are ionized and how many free electrons result — is the **equation of state**, and it is where Lecture 2 begins. With $n_e$ and the level populations in hand, we can compute opacity, and the spectrum comes into view.""")

md(r"""## Summary

- A synthetic spectrum is the forward problem $(T_{\rm eff}, \log g, \text{abundances}) \to F_\lambda$, split into a **model atmosphere** and **spectral synthesis**.
- We work in **CGS** with the reference's exact constants, and check every step against precomputed reference values, targeting $\lesssim10^{-6}$ relative deviation.
- The **Planck function** $B_\nu = (2h/c^2)\nu^3/(e^{h\nu/kT}-1)$ is the LTE source function; the overflow-safe Kurucz form reproduces the reference to machine precision.
- **Optical depth** $d\tau=\kappa\rho\,dz$ is the depth coordinate; the emergent flux forms near $\tau\approx2/3$, where $T\approx T_{\rm eff}$.
- The **grey atmosphere** gives $T(\tau)=T_{\rm eff}[\tfrac34(\tau+q(\tau))]^{1/4}$ and, with the cold-start $\kappa\equiv1$, $P_{\rm total}=g\tau$ and $\rho x=\tau$ — a complete first atmosphere, matched to the reference.""")

md(r"""## Practice exercises

**1. Wien's law from the code.** Using `planck_nu`, compute $B_\lambda(T)$ on a fine wavelength grid for $T = 3000, 5770, 10000\ \mathrm{K}$ and find the peak wavelength numerically. Verify that $\lambda_{\rm peak}\,T$ is constant and recover the constant $2.898\times10^{6}\ \mathrm{nm\,K}$. Which way does the solar peak move relative to our $500$–$510\ \mathrm{nm}$ window as the star cools?

**2. The Eddington–Barbier depth.** The grey result says $T(2/3)\approx T_{\rm eff}$. Evaluate `grey_temperature(5770, tau)` at $\tau = 1/2, 2/3, 1$ and tabulate $T/T_{\rm eff}$. By how many percent does the photospheric temperature change across that range, and what does that imply for how sharply lines of different strength sample temperature?

**3. A hotter, lower-gravity star.** Rebuild the grey atmosphere for an A-type dwarf, $T_{\rm eff}=9000\ \mathrm{K}$, $\log g = 4.0$, and overplot its $T(\tau)$ and $P_{\rm gas}(\tau)$ on the solar ones. Why is the gas pressure at fixed $\tau$ lower than the Sun's, given $P_{\rm total}=g\tau$? (You can check your structure by extending `_pipeline/make_references.py` for the new parameters.)""")

md(r"""## Further reading

- **Gray, D. F. (2005). *The Observation and Analysis of Stellar Photospheres*, 3rd ed., Cambridge University Press.** The standard text; Chapters 7–9 cover the grey atmosphere, optical depth, and the source function at the level of this course.
- **Mihalas, D. (1978). *Stellar Atmospheres*, 2nd ed., Freeman.** The rigorous reference for the transfer equation, the Hopf function, and the Eddington approximation.
- **Hubeny, I. & Mihalas, D. (2014). *Theory of Stellar Atmospheres*, Princeton University Press.** The modern comprehensive treatment, including LTE versus non-LTE.
- **Kurucz, R. L. (1970). *ATLAS: A Computer Program for Calculating Model Stellar Atmospheres*, SAO Special Report 309.** The original ATLAS, including the grey-start temperature structure we reproduced.
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz: A Pure-Python Reimplementation of Kurucz ATLAS12 and SYNTHE*](https://arxiv.org/abs/2603.11693).** The implementation our reference values are computed with.""")

# ── write ───────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
