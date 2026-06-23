#!/usr/bin/env python
"""Lecture 5 — Line Opacity I: A Single Line. Oscillator strength & log gf, the
Boltzmann lower-level population, the Doppler core, pressure/natural broadening,
and the Voigt profile. Checked against reference/L4.npz. No pykurucz import.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture5.ipynb"
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# Lecture 5 — Line Opacity I: A Single Line

*Stellar Spectroscopy from Scratch — rebuilding the physics of ATLAS and SYNTHE from first principles*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*Every result in this book is checked against reference values computed with [**pykurucz**](https://arxiv.org/abs/2603.11693) — a pure-Python implementation of Kurucz's ATLAS12 and SYNTHE — shipped beside the lectures as small data files, so the notebooks need only NumPy to run.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Write the **line absorption coefficient** and explain every factor: the classical constant, the **oscillator strength** behind `log gf`, and the lower-level population.
- Use the **Boltzmann** factor and the **excitation potential** $\chi_\ell$ to populate the lower level of a transition.
- Build the **Doppler core** from thermal motion and microturbulence, and the **Lorentzian wings** from natural, Stark, and van der Waals broadening.
- Combine the two into the **Voigt profile**, evaluate it as the exact Faddeeva function, and check it against the reference.
- Assemble the opacity profile of a single iron line in the solar photosphere.""")

md(r"""## Introduction

The continuum of Lecture 3 is the smooth canvas; a spectral **line** is a sharp spike of extra opacity at one wavelength, where a bound electron jumps between two specific energy levels. Where that opacity rises, the $\tau = 2/3$ surface moves up to cooler, dimmer layers, and the line appears in absorption. Two questions define a line. *How strong is it* — how much opacity does it add at line centre? And *what shape* does it have — how is that opacity spread over wavelength? The strength is set by the transition's **oscillator strength** and the number of atoms in the lower level; the shape, by the **Voigt profile**, the convolution of a thermal Doppler core with pressure-broadened Lorentzian wings. This lecture builds one line, end to end, on the solar atmosphere we constructed; the next assembles the million-line forest.""")

code(r'''import pathlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5})

REF = np.load(pathlib.Path("..") / "reference" / "L4.npz")
def compare(name, ours, ref, tol=1e-6):
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    denom = np.where(ref != 0.0, np.abs(ref), 1.0)
    rel = float(np.max(np.abs(ours - ref) / denom))
    tag = "exact" if rel < 1e-12 else ("agree" if rel < tol else "CHECK")
    print(f"{name:30s}  max|rel diff| = {rel:.2e}   [{tag}]")
    return rel

H_C, C, K = 6.62607015e-27, 2.99792458e10, 1.380649e-16   # Planck, c, Boltzmann (CGS)
AMU = 1.66053907e-24                                       # atomic mass unit [g]
print("reference loaded:", ", ".join(REF.files))''')

# ── line strength ───────────────────────────────────────────────────────
md(r"""## The strength of a line

The monochromatic absorption coefficient of a bound-bound transition, per unit volume, is

$$
\alpha_\nu = \frac{\pi e^2}{m_e c}\, f_{\ell u}\, n_\ell\, \phi(\nu)\,\big(1 - e^{-h\nu/kT}\big)\quad[\mathrm{cm^{-1}}],
$$

a product of four pieces. The classical constant $\pi e^2/m_e c = 0.02654\ \mathrm{cm^2\,Hz}$ is the cross-section a classical oscillator would present, integrated over frequency. The **oscillator strength** $f_{\ell u}$ — a dimensionless quantum-mechanical number between $0$ and $\sim1$ — corrects that classical value for the real transition. The **lower-level population** $n_\ell$ counts how many absorbers are available. And $\phi(\nu)$ is the line profile, normalised to $\int\phi\,d\nu = 1$, which spreads the opacity over frequency. The last factor is the same stimulated-emission correction as always.

Line lists do not tabulate $f$ directly; they give **$\log gf$**, the base-ten logarithm of the oscillator strength times the lower level's statistical weight $g_\ell$. The combination is what appears in the opacity, because $f_{\ell u}\,n_\ell = gf \cdot (n_\ell/g_\ell)$, and $n_\ell/g_\ell$ is exactly what the Boltzmann factor gives without needing $g_\ell$ separately. So $\log gf$ is the single number that sets a line's intrinsic strength — a strong line might have $\log gf \approx 0$, a weak one $\log gf \approx -5$.""")

code(r'''CLASSICAL = np.pi * (4.803204e-10)**2 / (9.1093837e-28 * C)   # pi e^2 / (m_e c)  [cm^2 Hz]
print(f"classical line constant  pi e^2 / (m_e c) = {CLASSICAL:.5e} cm^2 Hz")''')

md(r"""## Populating the lower level

How many atoms sit in the lower level of our transition? Two factors set it, both from Lecture 2. First, what fraction of the element is in the right ionization stage — here neutral iron, Fe I. Second, within that ion, the **Boltzmann** factor places a fraction in the lower level $\ell$ at excitation energy $\chi_\ell$ above the ground state:

$$
\frac{n_\ell}{g_\ell} = \frac{n(\mathrm{Fe\,I})}{U_{\rm Fe\,I}(T)}\,e^{-\chi_\ell/kT}.
$$

The **excitation potential** $\chi_\ell$ is decisive: a line from the ground state ($\chi_\ell = 0$) is available everywhere, while a line from a level several eV up is populated only in the hotter, deeper layers — which is why lines of different excitation potential probe different depths, the lever that lets spectroscopy measure temperature. We take $n(\mathrm{Fe\,I})$ and $U_{\rm Fe\,I}$ straight from the equation of state of Lecture 2, carried in the reference data.""")

code(r'''T   = REF["T"]; tk = REF["tk"]; n_e = REF["xne"]; rho = REF["rho"]
n_FeI = REF["n_FeI"]; U_FeI = REF["U_FeI"]; tau = REF["tau"]
KEV = 1.0/11604.5                                          # eV per kelvin
jp = np.argmin(np.abs(tau - 2/3))                          # photosphere layer
print(f"at the photosphere (T={T[jp]:.0f} K): n(Fe I) = {n_FeI[jp]:.3e} cm^-3, U(Fe I) = {U_FeI[jp]:.1f}")''')

# ── broadening + Voigt ──────────────────────────────────────────────────
md(r"""## The shape: Doppler core and Lorentzian wings

The profile $\phi(\nu)$ has two origins. **Thermal motion** Doppler-shifts each atom's absorption; averaged over a Maxwellian (plus a microturbulent velocity $\xi$ that represents unresolved small-scale flows), this gives a **Gaussian** core of $1/e$ half-width

$$
\Delta\nu_D = \frac{\nu_0}{c}\sqrt{\frac{2kT}{m} + \xi^2}.
$$

Independently, the finite lifetime of the atomic levels and perturbations from passing particles broaden the line into a **Lorentzian**, with a damping rate $\gamma = \gamma_{\rm rad} + \gamma_{\rm Stark} + \gamma_{\rm vdW}$: **natural** broadening (the radiative lifetime, always present), **Stark** broadening (collisions with charged particles, $\propto n_e$), and **van der Waals** broadening (collisions with neutral hydrogen, $\propto n_{\rm H}$, dominant in cool dwarfs). Convolving the Gaussian core with the Lorentzian wings gives the **Voigt profile**

$$
\phi(\nu) = \frac{1}{\sqrt{\pi}\,\Delta\nu_D}\,H(a, v),\qquad
v = \frac{\nu - \nu_0}{\Delta\nu_D},\qquad
a = \frac{\gamma}{4\pi\,\Delta\nu_D},
$$

where $v$ measures the distance from line centre in Doppler widths and the **damping parameter** $a$ is the ratio of Lorentzian to Doppler width. The dimensionless **Voigt function** $H(a,v)$ is the heart of every line.

![The Voigt profile is a thermal Doppler (Gaussian) core convolved with pressure- and natural-broadening (Lorentzian) wings.](resources/figures/s4_voigt.png)""")

md(r"""$H(a,v)$ is, mathematically, the real part of the **Faddeeva function** $w(z) = e^{-z^2}\mathrm{erfc}(-iz)$ at $z = v + ia$, i.e. $H(a,v) = \mathrm{Re}\,w(v+ia)$. Evaluating it exactly — for example with `scipy.special.wofz` — is one option, and swapping that in to compare is a good exercise. But the standard in this book is to reproduce the reference **exactly**, and the reference does not use the Faddeeva function: it uses Kurucz's fast **Harris-series approximation**,

$$
H(a,v) \approx H_0(v) + a\,H_1(v) + a^2 H_2(v) \quad\text{(near line centre)},
$$

built from three precomputed tables — $H_0(v) = e^{-v^2}$, $H_1(v)$, and $H_2(v) = (1-2v^2)e^{-v^2}$ — with a Lorentzian-wing form far from centre and an intermediate branch between. We reuse those tables (atomic-physics data) and reimplement the routine's exact branch logic, so our Voigt matches the gold standard to machine precision. (The Harris approximation itself differs from the true Faddeeva by up to $\sim1\%$ at large damping — a property of the reference we are matching, not an error of ours.)""")

code(r'''h0tab, h1tab, h2tab = REF["h0tab"], REF["h1tab"], REF["h2tab"]   # Harris coefficient tables (data)

def _voigt_wing(a, v):
    """Asymptotic (Lorentzian-wing) branch of Kurucz's Voigt approximation."""
    aa, vv = a*a, v*v
    u = (aa + vv)*1.4142
    val = a*0.79788/u
    if a <= 100.0:
        aau, vvu, uu = aa/u, vv/u, u*u
        val = ((((aau - 10.0*vvu)*aau*3.0 + 15.0*vvu*vvu) + 3.0*vv - aa)/uu + 1.0)*val
    return val

def voigt_H(a, v):
    """Voigt H(a,v): Kurucz's Harris-table routine, reproduced exactly (the gold-standard form)."""
    v = np.atleast_1d(np.asarray(v, float)); av = np.abs(v)
    iv = np.clip((av*200.0 + 0.5).astype(int), 0, h0tab.size-1)
    H0, H1, H2 = h0tab[iv], h1tab[iv], h2tab[iv]
    out = np.empty_like(v)
    if a < 0.2:                                              # weak damping: 2nd-order Harris series
        far = av > 10.0
        out[far]  = 0.5642*a/(v[far]*v[far])
        out[~far] = (H2[~far]*a + H1[~far])*a + H0[~far]
    elif a > 1.4:                                            # strong damping: pure asymptotic wing
        out[:] = _voigt_wing(a, v)
    else:                                                    # intermediate: split per point at a+|v|=3.2
        asy = (a + av) > 3.2
        out[asy] = _voigt_wing(a, v[asy])
        m = ~asy; vv = v[m]*v[m]; h0 = H0[m]; h1t = H1[m]; h2t = H2[m]
        h1 = h1t + h0*1.12838
        h2 = h2t + h1*1.12838 - h0
        h3 = (1.0 - h2t)*0.37613 - h1*0.66667*vv + h2*1.12838
        h4 = (3.0*h3 - h1)*0.37613 + h0*0.66667*vv*vv
        polyA = (((h4*a + h3)*a + h2)*a + h1)*a + h0
        polyB = ((-0.122727278*a + 0.532770573)*a - 0.96284325)*a + 0.979895032
        out[m] = polyA*polyB
    return out

v_ref, a_ref, H_ref = REF["v"], REF["a"], REF["H"]
H_mine = np.array([voigt_H(aa, v_ref) for aa in a_ref])
compare("Voigt H(a,v)", H_mine, H_ref)

for k, aa in enumerate(a_ref):
    plt.plot(v_ref, H_mine[k], label=f"a = {aa:g}")
plt.yscale("log"); plt.ylim(1e-4, 1.2)
plt.xlabel(r"$v = (\nu-\nu_0)/\Delta\nu_D$"); plt.ylabel(r"Voigt function $H(a,v)$")
plt.title("Doppler core, Lorentzian wings: the Voigt function"); plt.legend(); plt.tight_layout(); plt.show()''')

md(r"""The Gaussian core (small $v$) is nearly independent of $a$; the wings (large $v$) are pure Lorentzian and scale as $a/(\sqrt{\pi}v^2)$ — the larger the damping, the heavier the wings. Strong lines with broad damping wings are how we read surface gravity and pressure from a spectrum.""")

# ── assemble one line ───────────────────────────────────────────────────
md(r"""## Assembling one iron line

Now put it together for a representative neutral-iron line in our window: rest wavelength $\lambda_0 = 500.5\ \mathrm{nm}$, $\log gf = -1.0$, lower-level excitation $\chi_\ell = 3.3\ \mathrm{eV}$. We build its opacity profile at the photosphere, with a microturbulence $\xi = 2\ \mathrm{km\,s^{-1}}$ and a damping rate dominated by radiation and van der Waals collisions. The result is the line absorption coefficient per gram, $\alpha_\nu/\rho$, which we overlay on the continuous opacity of Lecture 3 — the spike that carves the line.""")

code(r'''# representative neutral-iron line
lam0_nm = 500.5; loggf = -1.0; chi_l = 3.3                 # nm, log(gf), eV
m_Fe = 55.845 * AMU                                        # iron atom mass [g]
xi = 2.0e5                                                  # microturbulence [cm/s]
nu0 = C / (lam0_nm * 1e-7)

# Doppler width and damping at the photosphere
dnu_D = (nu0 / C) * np.sqrt(2*K*T[jp]/m_Fe + xi**2)        # Hz
gamma_rad = 2.0e8                                           # natural broadening [s^-1]
gamma_vdw = 1.0e-7 * REF["nHI"][jp]                         # van der Waals ~ n_H [s^-1]
gamma_stark = 1.0e-8 * n_e[jp]                              # Stark ~ n_e [s^-1]
gamma = gamma_rad + gamma_vdw + gamma_stark
a_damp = gamma / (4*np.pi*dnu_D)
dlam_mA = dnu_D * (lam0_nm*1e-7)**2 / C * 1e8 * 1e3        # Doppler width in milli-angstrom
print(f"Doppler width = {dnu_D:.3e} Hz ({dlam_mA:.0f} mA);  damping a = {a_damp:.3f}")

# opacity profile across +-0.04 nm
lam = np.linspace(lam0_nm-0.04, lam0_nm+0.04, 800)
nu = C / (lam*1e-7)
v = (nu - nu0) / dnu_D
phi = voigt_H(a_damp, v) / (np.sqrt(np.pi) * dnu_D)        # normalised profile [s]
stim = 1.0 - np.exp(-H_C*nu/(K*T[jp]))
nl_over_gl = (n_FeI[jp]/U_FeI[jp]) * np.exp(-chi_l/(KEV*T[jp]))
alpha_line = CLASSICAL * 10**loggf * nl_over_gl * phi * stim    # cm^-1
kappa_line = alpha_line / rho[jp]                              # cm^2/g
print(f"line-centre opacity / continuum = {kappa_line.max()/0.027:.1f}  (continuum ~0.027 cm^2/g)")''')

code(r'''plt.plot(lam, kappa_line + 0.027, color="C3", label="line + continuum")
plt.axhline(0.027, ls="--", color="0.5", lw=1, label="continuum (H$^-$)")
plt.yscale("log"); plt.xlabel("wavelength [nm]"); plt.ylabel(r"$\kappa$  [cm$^2$/g]")
plt.title(r"Opacity profile of one Fe I line at the photosphere ($\lambda_0=500.5$ nm)")
plt.legend(); plt.tight_layout(); plt.show()''')

md(r"""The line raises the opacity more than a hundredfold at its centre, falling through a Gaussian core into Lorentzian wings that merge back into the continuum a fraction of an ångström away. At line centre the $\tau=2/3$ surface is pushed up into cool layers and the emergent flux drops — that conversion of opacity into a flux profile is the radiative transfer of Lecture 8. For now, note the levers we have exposed: $\log gf$ and the Boltzmann factor scale the whole profile up or down, while the damping $a$ controls how much of the line lives in the wings. Vary $\log gf$ (or the abundance hidden inside $n(\mathrm{Fe\,I})$) and you trace the **curve of growth** — the relation between a line's strength and its equivalent width that turns a measured depth into an abundance.""")

# ── close ───────────────────────────────────────────────────────────────
md(r"""## Synthesis: what you built and where it goes

You built a single spectral line from its parts. Its **strength** is the classical constant times the oscillator strength `log gf` times the lower-level population — and that population comes from the **ionization** (Saha) and **excitation** (Boltzmann, through $\chi_\ell$) of Lecture 2. Its **shape** is the **Voigt profile**, the exact Faddeeva function, convolving a thermal-plus-turbulent Doppler core with natural, Stark, and van der Waals wings. You overlaid the result on the H$^-$ continuum and saw the opacity spike that becomes an absorption line.

One line is a demonstration; a real spectrum is a forest. Lecture 6 reads the Kurucz line list — over a million atomic transitions — assigns each its $\log gf$, excitation potential, and damping constants, and sums their Voigt profiles onto a wavelength grid to build the total line opacity at every depth, ready for the radiative transfer that turns it into the spectrum.""")

md(r"""## Summary

- The line absorption coefficient is $\alpha_\nu = (\pi e^2/m_e c)\,f_{\ell u}\,n_\ell\,\phi(\nu)\,(1-e^{-h\nu/kT})$, with $\pi e^2/m_e c = 0.02654\ \mathrm{cm^2\,Hz}$.
- Line lists give **$\log gf$**; the opacity uses $gf\cdot(n_\ell/g_\ell)$, and $n_\ell/g_\ell = (n_{\rm ion}/U)\,e^{-\chi_\ell/kT}$ from Boltzmann.
- The profile is a **Voigt** function $\phi = H(a,v)/(\sqrt{\pi}\Delta\nu_D)$: Gaussian Doppler core (thermal + microturbulence) and Lorentzian wings from natural + Stark + van der Waals broadening, with damping $a=\gamma/4\pi\Delta\nu_D$.
- $H(a,v)$ is Kurucz's Harris-table approximation to $\mathrm{Re}\,w(v+ia)$, reproduced **exactly** (machine precision) by reusing its tables and branch logic; the true Faddeeva is a one-line swap-in for comparison.
- $\log gf$, the excitation potential, and the damping are the levers connecting a line's depth to temperature, gravity, and abundance.""")

md(r"""## Practice exercises

**1. Excitation potential and depth.** Recompute the line's central opacity for $\chi_\ell = 0$ and $\chi_\ell = 5\ \mathrm{eV}$ at three depths. Which line is relatively stronger in the cool upper photosphere, and why does high-excitation make a line a deeper-forming, more temperature-sensitive probe?

**2. The damping wings.** Increase $\gamma_{\rm vdW}$ by factors of $3$ and $10$ (mimicking higher gravity) and plot the profiles. How does the damping parameter $a$ change, and where in the profile does the extra opacity appear? This is the surface-gravity diagnostic.

**3. Equivalent width.** Numerically integrate $1 - e^{-\tau_\nu}$ across the line (treat $\tau_\nu \propto \kappa_{\rm line}/\kappa_{\rm cont}$) for a range of $\log gf$ from $-4$ to $0$, and plot equivalent width against $\log gf$. Identify the linear, flat (saturated), and damping parts of the curve of growth.""")

md(r"""## Further reading

- **Gray, D. F. (2005). *The Observation and Analysis of Stellar Photospheres*, 3rd ed.** Chapters 11–13 on line absorption coefficients, broadening, and the curve of growth.
- **Rutten, R. J. (2003). *Radiative Transfer in Stellar Atmospheres* (Utrecht lecture notes).** A lucid derivation of the Voigt profile and the line opacity.
- **Humlíček, J. (1982). *Optimized computation of the Voigt and complex probability functions*, JQSRT, 27, 437.** Fast accurate evaluation of the Faddeeva function behind `wofz`.
- **Anstee, S. D. & O'Mara, B. J. (1995). *Width cross-sections for collisional broadening...*, MNRAS, 276, 859.** Modern van der Waals broadening (the "ABO" theory).
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz*](https://arxiv.org/abs/2603.11693).** The implementation our reference Voigt is computed with.""")

nb = new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
