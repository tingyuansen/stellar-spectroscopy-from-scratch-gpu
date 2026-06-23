#!/usr/bin/env python
"""Assemble content/Lecture3.ipynb. Lecture 3 — Continuous Opacity:
H- bound-free & free-free (John 1988), Rayleigh + Thomson scattering, the smooth
continuum the lines sit on. Checked against reference/L3.npz. No pykurucz import.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture3.ipynb"
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# Lecture 3 — Continuous Opacity

*Stellar Spectroscopy from Scratch — rebuilding the physics of ATLAS and SYNTHE from first principles*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*Every result in this book is checked against reference values computed with [**pykurucz**](https://arxiv.org/abs/2603.11693) — a pure-Python implementation of Kurucz's ATLAS12 and SYNTHE — shipped beside the lectures as small data files, so the notebooks need only NumPy to run.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Distinguish **true absorption** from **scattering**, and write the continuous extinction coefficient per gram.
- Explain why the **negative hydrogen ion H$^-$** dominates the visible continuum of a cool star, and compute its abundance from the Saha equation.
- Evaluate the **H$^-$ bound-free and free-free** opacity with the standard analytic fits and reproduce the reference continuum to a few percent.
- Show that the photospheric **scattering** is Rayleigh scattering off neutral hydrogen, not Thomson scattering off electrons.
- Build the total continuous opacity $\kappa_\lambda$ at every depth and locate where the continuum is formed.""")

md(r"""## Introduction

With the equation of state in hand we can finally compute an opacity. Opacity comes in two flavours, and the distinction matters for radiative transfer later. **True absorption** destroys a photon and converts it to thermal energy — the gas re-emits according to its temperature ($S_\lambda = B_\lambda$). **Scattering** merely redirects a photon without thermalising it — the source function then depends on the radiation field itself. This lecture builds the **continuous** opacity: the smooth, slowly-varying background that sets the overall brightness of the star and the floor from which the sharp spectral lines are carved.

In a cool star like the Sun the dominant continuous absorber is one of the more surprising species in astrophysics: the **negative hydrogen ion, H$^-$** — a neutral hydrogen atom that has captured a *second*, very loosely bound electron. It binds that electron by only $0.754\ \mathrm{eV}$, so it is fragile and rare, but neutral hydrogen is so overwhelmingly abundant, and the free electrons supplied by the metals (Lecture 2) so available, that H$^-$ swamps every other continuum source across the optical. We build it, the scattering terms, and the total — and check each against the reference.""")

md(r"""Load the reference continuum and the atmosphere it was built on.""")

code(r'''import pathlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5})

REF = np.load(pathlib.Path("..") / "reference" / "L3.npz")
def compare(name, ours, ref, tol=1e-6):
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    denom = np.where(ref != 0.0, np.abs(ref), 1.0)
    rel = float(np.max(np.abs(ours - ref) / denom))
    tag = "exact" if rel < 1e-12 else ("agree" if rel < tol else "CHECK")
    print(f"{name:30s}  max|rel diff| = {rel:.2e}   [{tag}]")
    return rel

wl  = REF["wl"]                                   # wavelength grid [nm]
T   = REF["T"][:, None]                            # temperature [K], shaped for broadcasting over wl
n_e = REF["n_e"][:, None]                          # electron density [cm^-3]
rho = REF["rho"][:, None]                          # mass density [g cm^-3]
nHI = REF["nHI"][:, None]                          # neutral hydrogen number density [cm^-3]
tau, rhox = REF["tau"], REF["rhox"]
H, C, K = 6.62607015e-27, 2.99792458e10, 1.380649e-16
KEV, SAHA = 1.0/11604.5, 2.4148e15
nu = C / (wl[None, :] * 1e-7)                       # frequency [Hz] at each wavelength
print(f"continuum grid: {REF['absorption'].shape[0]} layers x {wl.size} wavelengths, "
      f"{wl[0]:.0f}-{wl[-1]:.0f} nm")''')

md(r"""## The stimulated-emission factor

Every true-absorption coefficient carries a correction we met in Lecture 1. In the radiation field, some photons *stimulate* a bound electron to emit rather than absorb; the net absorption is the difference. For a process in LTE the correction multiplies the raw cross-section by

$$
\big(1 - e^{-h\nu/kT}\big),
$$

the same $1-e^{-x}$ that sat in the denominator of the Planck function. It is near $1$ in the blue and drops toward the infrared, where $h\nu \lesssim kT$. Scattering, which does not exchange energy with the gas, carries no such factor.""")

code(r'''stim = 1.0 - np.exp(-H * nu / (K * T))             # stimulated-emission correction, shape (layers, wl)
print(f"stimulated-emission factor at 505 nm: {stim[:,100].min():.3f} (hot, deep) .. {stim[:,100].max():.3f} (cool)")''')

md(r"""## How much H$^-$ is there? The Saha balance

H$^-$ forms and is destroyed by the reaction $\mathrm{H} + e^- \rightleftharpoons \mathrm{H}^-$, so its abundance follows a Saha equation exactly like the ones in Lecture 2, but with the tiny binding energy $\chi = 0.754\ \mathrm{eV}$ in place of an ionization potential:

$$
\frac{n(\mathrm{H\,I})\,n_e}{n(\mathrm{H}^-)}
= \frac{2\,g_{\rm H\,I}}{g_{\mathrm{H}^-}}
\left(\frac{2\pi m_e kT}{h^2}\right)^{3/2} e^{-\chi/kT},
$$

with statistical weights $g_{\rm H\,I}=2$ (the hydrogen ground state) and $g_{\mathrm{H}^-}=1$ (the closed-shell ion), so the leading factor is $2\cdot2/1 = 4$. Solving for the H$^-$ density,

$$
n(\mathrm{H}^-) = \frac{n(\mathrm{H\,I})\,n_e}{4\,(2.4148\times10^{15})\,T^{3/2}}\;e^{+\chi/kT}.
$$

The positive exponent says H$^-$ is *favoured* at low temperature — the captured electron is more easily held when there is less thermal energy to knock it loose. And the explicit $n_e$ is why H$^-$ opacity tracks the electron density, and hence the metal abundance, that Lecture 2 worked out.

![The H$^-$ ion — a hydrogen atom holding a second, weakly bound electron (0.754 eV) — is the dominant continuous absorber in cool stars; a photon detaches the electron, and its abundance tracks the electron density.](resources/figures/s3_hminus.png)""")

code(r'''chi_Hminus = 0.754                                  # H- electron binding energy [eV]
n_Hminus = nHI * n_e * np.exp(chi_Hminus / (KEV * T)) / (4.0 * SAHA * T**1.5)   # [cm^-3]
print(f"n(H-)/n(H I) at the photosphere: {(n_Hminus/nHI)[50,0]:.2e}  "
      f"(about two H- per billion H atoms)")''')

md(r"""## H$^-$ bound-free and free-free opacity

H$^-$ absorbs by two channels. **Bound-free** (photodetachment) ejects the bound electron, $\mathrm{H}^- + \gamma \to \mathrm{H} + e^-$; it has a threshold at $\lambda_0 = 1.6419\ \mathrm{\mu m}$ (corresponding to $0.754\ \mathrm{eV}$) and peaks in the red. **Free-free** is absorption by a passing electron in the field of a neutral hydrogen atom, $\mathrm{H} + e^- + \gamma \to \mathrm{H} + e^-$; it has no threshold and rises into the infrared.

For both we use the standard analytic fits of **John (1988)**, accurate to $\sim1\%$ and used throughout stellar-atmosphere work. The bound-free photodetachment cross-section, with $\lambda$ in microns and $f \equiv 1/\lambda - 1/\lambda_0$, is

$$
\sigma_{\rm bf}(\lambda) = 10^{-18}\,\lambda^3\,f^{3/2}\sum_{n=0}^{5} C_n\, f^{n/2}\quad[\mathrm{cm^2}],
$$

and the opacity per gram is $\kappa_{\rm bf} = n(\mathrm{H}^-)\,\sigma_{\rm bf}\,(1-e^{-h\nu/kT})/\rho$. The free-free coefficient is a temperature- and wavelength-polynomial in $\theta = 5040/T$, returning the absorption per neutral H atom per unit electron pressure $P_e = n_e kT$.""")

code(r'''# John (1988) H- bound-free photodetachment cross-section
lam_um = wl[None, :] * 1e-3                          # wavelength in microns
lam0 = 1.6419                                        # threshold [um] (0.754 eV)
f = 1.0/lam_um - 1.0/lam0
C_bf = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982]
poly = sum(C_bf[n] * f**(n/2.0) for n in range(6))
sigma_bf = np.where(lam_um < lam0, 1e-18 * lam_um**3 * f**1.5 * poly, 0.0)   # cm^2
kappa_bf = n_Hminus * sigma_bf * stim / rho          # cm^2/g

# John (1988) H- free-free coefficient (lambda > 0.3645 um branch), per H I atom per unit P_e
A=[0,2483.346,-3449.889,2200.040,-696.271,88.283]; B=[0,285.827,-1158.382,2427.719,-1841.400,444.517]
Cc=[0,-2054.291,8746.523,-13651.105,8624.970,-1863.864]; D=[0,2827.776,-11485.632,16755.524,-10051.530,2095.288]
E=[0,-1341.537,5303.609,-7510.494,4400.067,-901.788]; F=[0,208.952,-812.939,1132.738,-655.020,132.985]
theta = 5040.0 / T
kff = sum(theta**((n+1)/2.0) * (A[n]*lam_um**2 + B[n] + Cc[n]/lam_um + D[n]/lam_um**2
                                + E[n]/lam_um**3 + F[n]/lam_um**4) for n in range(1, 6))
P_e = n_e * K * T                                    # electron pressure [dyn cm^-2]
kappa_ff = 1e-29 * kff * P_e * nHI / rho             # cm^2/g

kappa_Hminus = kappa_bf + kappa_ff
# H- dominates only where the optical spectrum forms; deep, hot layers add H and metal
# bound-free that we are not modelling. Benchmark over the spectrum-forming layers.
form = (tau > 1e-3) & (tau < 3.0)
rel = np.abs(kappa_Hminus[form] - REF["absorption"][form]) / REF["absorption"][form]
print(f"continuum absorption (H-), spectrum-forming layers:  "
      f"median|rel diff| = {np.median(rel):.2e}   max = {np.max(rel):.2e}")''')

md(r"""About two percent through the layers where the optical spectrum forms — our textbook H$^-$ reproduces the reference absorption, the bound-free channel carrying most of it at these wavelengths. Two honest caveats. The residual is the detailed photodetachment table the production code carries. And we restrict the comparison to the spectrum-forming layers on purpose: far below the photosphere (deeper than $\tau_{\rm Ross}\sim 5$, hotter than $\sim10{,}000\ \mathrm{K}$) hydrogen and metal bound-free edges take over from H$^-$, but those layers are never seen in the emergent optical spectrum, so we do not model them here. Where it matters, H$^-$ alone carries the continuous absorption of the solar photosphere.""")

md(r"""## Scattering: Rayleigh beats Thomson

Now the scattering. The textbook reflex is **Thomson scattering** off free electrons, with the constant cross-section $\sigma_T = 0.6653\times10^{-24}\ \mathrm{cm^2}$. But in a cool photosphere the free electrons are scarce ($n_e \sim 10^{13}\ \mathrm{cm^{-3}}$, Lecture 2), while neutral hydrogen is everywhere. **Rayleigh scattering** off the bound electrons of neutral hydrogen — the same $\lambda^{-4}$ process that makes the sky blue — therefore dominates. We use the standard polarizability fit (Dalgarno), with $\lambda$ in ångström,

$$
\sigma_{\rm Ray}(\lambda) = \frac{5.799\times10^{-13}}{\lambda^4} + \frac{1.422\times10^{-6}}{\lambda^6} + \frac{2.784}{\lambda^8}\quad[\mathrm{cm^2}],
$$

and add Thomson for completeness. Neither carries a stimulated-emission factor.""")

code(r'''lamA = wl[None, :] * 10.0                            # wavelength in angstrom
sigma_Ray = 5.799e-13/lamA**4 + 1.422e-6/lamA**6 + 2.784/lamA**8   # cm^2 per H I atom
kappa_Ray = sigma_Ray * nHI / rho                    # cm^2/g
kappa_Thomson = 0.6653e-24 * n_e / rho * np.ones_like(nu)
kappa_scat = kappa_Ray + kappa_Thomson
compare("scattering (Rayleigh+Thomson)", kappa_scat, REF["scattering"], tol=5e-2)
print(f"at the photosphere, Rayleigh is {kappa_Ray[50,100]/kappa_Thomson[50,100]:.0f}x Thomson; "
      f"scattering is {100*kappa_scat[50,100]/kappa_Hminus[50,100]:.0f}% of absorption")''')

md(r"""Rayleigh outweighs Thomson roughly tenfold here, and scattering as a whole is only a couple of percent of the H$^-$ absorption — the solar optical continuum is an *absorption* continuum, which is why its source function is so close to the Planck function and the emergent flux so close to a blackbody shaped by H$^-$. In the ultraviolet, where $\lambda^{-4}$ Rayleigh climbs and metal photoionization switches on, the balance shifts; in hot stars, where hydrogen is ionized, Thomson takes over.""")

md(r"""## The total continuum and where it forms

Add absorption and scattering for the total continuous extinction, and check it against the reference. Then convert it to optical depth: since $d\tau_\lambda = \kappa_\lambda\,\rho\,dz = \kappa_\lambda\,d(\rho x)$, integrating the continuum opacity over the column mass tells us the depth from which the continuum escapes — and confirms the Eddington–Barbier picture of Lecture 1.""")

code(r'''kappa_total = kappa_Hminus + kappa_scat
ref_total = REF["absorption"] + REF["scattering"]
rel = np.abs(kappa_total[form] - ref_total[form]) / ref_total[form]
print(f"total continuum, spectrum-forming layers:  "
      f"median|rel diff| = {np.median(rel):.2e}   max = {np.max(rel):.2e}")

# continuum optical depth at mid-window (505 nm), integrated over column mass
k505 = kappa_total[:, 100]
tau_cont = np.zeros_like(rhox)
tau_cont[1:] = np.cumsum(0.5*(k505[1:]+k505[:-1]) * np.diff(rhox))
j23 = np.argmin(np.abs(tau_cont - 2/3))
print(f"the 505 nm continuum reaches tau=2/3 at T = {REF['T'][j23]:.0f} K  "
      f"(log tau_Ross = {np.log10(tau[j23]):.2f})")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))
ax[0].plot(np.log10(tau), np.log10(kappa_total[:,100]), color="C0", label="total")
ax[0].plot(np.log10(tau), np.log10(kappa_Hminus[:,100]), "--", color="C3", label="H$^-$ absorption")
ax[0].plot(np.log10(tau), np.log10(kappa_scat[:,100]), ":", color="C2", label="scattering")
ax[0].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[0].set_ylabel(r"$\log_{10}\kappa_{505}$  [cm$^2$/g]")
ax[0].set_title("Continuum opacity vs depth"); ax[0].legend()
ax[1].plot(np.log10(tau), tau_cont, color="C0"); ax[1].axhline(2/3, ls="--", color="0.5", lw=1)
ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$")
ax[1].set_ylabel(r"continuum optical depth $\tau_{505}$"); ax[1].set_title("Where the continuum forms")
fig.tight_layout(); plt.show()''')

md(r"""The total continuum matches the reference to about two percent through the spectrum-forming layers — the level at which clean textbook opacity reproduces a production code, the residual being the detailed cross-section tables it carries. As in the equation of state, we adopt the reference continuum for the bit-identical spectrum of Lecture 8; here we have shown the physics is right. The right-hand panel locates the continuum's origin: at $505\ \mathrm{nm}$ it forms around $T\approx6700\ \mathrm{K}$ — close to the real solar value, and somewhat hotter than the grey $\tau_{\rm Ross}=2/3$ photosphere ($5800\ \mathrm{K}$). That offset is a symptom of the crude grey temperature structure and its placeholder Rosseland opacity ($\kappa\equiv1$); Lectures 10–11 rebuild both self-consistently and the depth scales fall into line. The point stands: the continuum opacity sets which layer — and so which temperature — the brightness reflects.""")

md(r"""## Synthesis: what you built and where it goes

You turned the equation of state into an opacity. You found that the visible continuum of a cool star is carried by the **H$^-$ ion**, whose abundance you computed from a Saha balance and whose **bound-free and free-free** absorption you evaluated with the John (1988) fits, reproducing the reference to a few percent. You saw that photospheric **scattering is Rayleigh off neutral hydrogen**, not Thomson, and that scattering is a minor correction to a fundamentally absorptive continuum. And you confirmed that this continuum forms at $\tau\approx2/3$, tying the opacity back to the grey temperature structure.

This smooth continuum is the canvas. In Lecture 5 we paint the first spectral line onto it — a sharp, deep spike of extra opacity at one wavelength — and learn the profile that every line shares.""")

md(r"""## Summary

- Continuous **extinction = true absorption + scattering**; only absorption carries the stimulated-emission factor $1-e^{-h\nu/kT}$ and thermalises ($S=B$).
- **H$^-$** dominates the visible continuum of cool stars; its density follows a Saha balance with $\chi=0.754\ \mathrm{eV}$ and scales with $n_e$.
- The **John (1988)** bound-free and free-free fits reproduce the reference H$^-$ absorption to a few percent.
- Photospheric **scattering is Rayleigh off H I** ($\propto\lambda^{-4}$), about ten times Thomson and only $\sim2\%$ of the absorption.
- The $505\ \mathrm{nm}$ continuum forms at $T\approx6700\ \mathrm{K}$, somewhat hotter than the grey $\tau=2/3$ photosphere — a sign the crude grey structure needs the refinement of Part III.""")

md(r"""## Practice exercises

**1. The H$^-$ peak.** Plot $\sigma_{\rm bf}(\lambda)$ from the John fit across $0.2$–$1.7\ \mathrm{\mu m}$. Where does the bound-free cross-section peak, and what happens beyond the threshold $\lambda_0 = 1.6419\ \mathrm{\mu m}$? Why does the free-free term matter more in the infrared?

**2. Metal-poor continuum.** H$^-$ opacity scales with $n_e$, which (Lecture 2) scales with metal abundance. Recompute $\kappa_{\rm H^-}$ with $n_e$ reduced by a factor of ten and describe how the continuum brightness and the depth of the $\tau=2/3$ layer would change. Why are metal-poor stars said to have "more transparent" atmospheres?

**3. When does Thomson win?** Using the Saha equation of Lecture 2, estimate the temperature at which hydrogen becomes ionized enough that Thomson scattering overtakes Rayleigh at $500\ \mathrm{nm}$. What kind of star is that?""")

md(r"""## Further reading

- **John, T. L. (1988). [*Continuous absorption by the negative hydrogen ion reconsidered*](https://ui.adsabs.harvard.edu/abs/1988A%26A...193..189J/abstract), A&A, 193, 189.** The bound-free and free-free fits used here.
- **Gray, D. F. (2005). *The Observation and Analysis of Stellar Photospheres*, 3rd ed.** Chapter 8 on continuous opacity, with the H$^-$ and scattering terms worked through.
- **Wildt, R. (1939). *Negative Ions of Hydrogen and the Opacity of Stellar Atmospheres*, ApJ, 89, 295.** The paper that identified H$^-$ as the key opacity source.
- **Rutten, R. J. (2003). *Radiative Transfer in Stellar Atmospheres* (lecture notes, Utrecht).** A clear modern treatment of continuum formation and the Eddington–Barbier relation.
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz*](https://arxiv.org/abs/2603.11693).** The implementation our reference continuum is computed with.""")

nb = new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
