#!/usr/bin/env python
"""Lecture 4 — Line Opacity I: A Single Line. Oscillator strength & log gf, the
Boltzmann lower-level population, the Doppler core, pressure/natural broadening,
and the Voigt profile. Checked against reference/L4.npz. No pykurucz import.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture4.ipynb"
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# Lecture 4 — Line Opacity I: A Single Line

*Stellar Spectroscopy from Scratch — rebuilding the physics of ATLAS and SYNTHE from first principles*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*Every result in this book is checked against reference values computed with [**pykurucz**](https://arxiv.org/abs/2603.11693) — a pure-Python implementation of Kurucz's ATLAS12 and SYNTHE — shipped beside the lectures as small data files, so the notebooks need only NumPy to run.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Write the **line absorption coefficient** and explain every factor: the classical constant, the **oscillator strength** behind `log gf`, and the lower-level population.
- Use the **Boltzmann** factor and the **excitation potential** $\chi_\ell$ to populate the lower level of a transition.
- Build the **Doppler core** from thermal motion and microturbulence, and the **Lorentzian wings** from natural, Stark, and van der Waals broadening.
- Combine the two into the **Voigt profile** — mathematically the real part of the Faddeeva function — and reproduce Kurucz's Harris-table approximation to it, the form the reference uses.
- Assemble the opacity profile of a single iron line in the solar photosphere.""")

md(r"""## Introduction

The continuum of Lecture 3 is the smooth canvas; a spectral **line** is a sharp spike of extra opacity at one wavelength, where a bound electron jumps between two specific energy levels. Where that opacity rises, the $\tau = 2/3$ surface moves up to cooler, dimmer layers, and the line appears in absorption. Two questions define a line. *How strong is it* — how much opacity does it add at line centre? And *what shape* does it have — how is that opacity spread over wavelength?

The strength is set by the transition's **oscillator strength** and the number of atoms in the lower level; the shape, by the **Voigt profile**, the convolution of a thermal Doppler core with pressure-broadened Lorentzian wings. We will lean on results already in hand — the stimulated-emission factor from Lecture 1 and the Boltzmann lower-level population from Lecture 2 — and spend our effort on what is genuinely new here: the line strength carried by $\log gf$, the Voigt profile, and the damping budget that fills its wings. This lecture builds one line, end to end, on the solar atmosphere we constructed; the next assembles the million-line forest.""")

md(r"""## Setup

We import only NumPy (arithmetic) and Matplotlib (plots), load the reference bundle `reference/L4.npz` — the pre-computed pykurucz quantities every result is checked against — and define a `compare` helper that reports the worst relative difference between our arrays and the reference. All constants are in CGS throughout the book.""")

code(r'''import pathlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5})

# Reference values from pykurucz; every result below is checked against these arrays.
REF = np.load(pathlib.Path("..") / "reference" / "L4.npz")

def compare(name, ours, ref, tol=1e-6):
    """Report the worst relative difference; 'exact' = machine precision, 'agree' = within tol."""
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    denom = np.where(ref != 0.0, np.abs(ref), 1.0)          # avoid divide-by-zero where ref==0
    rel = float(np.max(np.abs(ours - ref) / denom))
    tag = "exact" if rel < 1e-12 else ("agree" if rel < tol else "CHECK")
    print(f"{name:30s}  max|rel diff| = {rel:.2e}   [{tag}]")
    return rel

# Physical constants in CGS units.
H_C, C, K = 6.62607015e-27, 2.99792458e10, 1.380649e-16   # Planck h, light speed c, Boltzmann k
AMU = 1.66053907e-24                                       # atomic mass unit [g]
print("reference loaded:", ", ".join(REF.files))''')

# ── line strength ───────────────────────────────────────────────────────
md(r"""## The strength of a line

The monochromatic absorption coefficient of a bound-bound transition, per unit volume, is

$$
\alpha_\nu = \frac{\pi e^2}{m_e c}\, f_{\ell u}\, n_\ell\, \phi(\nu)\,\big(1 - e^{-h\nu/kT}\big)\quad[\mathrm{cm^{-1}}],
$$

a product of four pieces. The classical constant $\pi e^2/m_e c = 0.02654\ \mathrm{cm^2\,Hz}$ is the cross-section a classical oscillator would present, integrated over frequency. The **oscillator strength** $f_{\ell u}$ — a dimensionless quantum-mechanical number, usually of order unity or smaller but not strictly bounded by 1 — corrects that classical value for the real transition. The **lower-level population** $n_\ell$ counts how many absorbers are available. And $\phi(\nu)$ is the line profile, normalised to $\int\phi\,d\nu = 1$, which spreads the opacity over frequency. The last factor is the same stimulated-emission correction $(1 - e^{-h\nu/kT})$ we met in Lecture 1; nothing new here.

Line lists do not tabulate $f$ directly; they give **$\log gf$**, the base-ten logarithm of the oscillator strength times the lower level's statistical weight $g_\ell$. The combination is what appears in the opacity, because $f_{\ell u}\,n_\ell = gf \cdot (n_\ell/g_\ell)$, and $n_\ell/g_\ell$ is exactly what the Boltzmann factor gives without needing $g_\ell$ separately. Thus the tabulated $gf$ is not a new physical factor; it is a bookkeeping device that pairs naturally with the Boltzmann population *per sublevel*, $n_\ell/g_\ell$. So $\log gf$ is the single number that sets a line's intrinsic strength — a strong line might have $\log gf \approx 0$, a weak one $\log gf \approx -5$.

The code cell below evaluates only the classical constant; the remaining factors are assembled later once we have the population and profile.""")

code(r'''CLASSICAL = np.pi * (4.803204e-10)**2 / (9.1093837e-28 * C)   # pi e^2 / (m_e c)  [cm^2 Hz]
print(f"classical line constant  pi e^2 / (m_e c) = {CLASSICAL:.5e} cm^2 Hz")''')

md(r"""## Populating the lower level

How many atoms sit in the lower level of our transition? Two factors set it, both from Lecture 2. First, what fraction of the element is in the right ionization stage — here neutral iron, Fe I. Second, within that ion, the **Boltzmann** factor places a fraction in the lower level $\ell$ at excitation energy $\chi_\ell$ above the ground state:

$$
\frac{n_\ell}{g_\ell} = \frac{n(\mathrm{Fe\,I})}{U_{\rm Fe\,I}(T)}\,e^{-\chi_\ell/kT}.
$$

The **excitation potential** $\chi_\ell$ is decisive: a line from the ground state ($\chi_\ell = 0$) is available everywhere, while a line from a level several eV up is populated only in the hotter, deeper layers — which is why lines of different excitation potential probe different depths, the lever that lets spectroscopy measure temperature. We take $n(\mathrm{Fe\,I})$ and $U_{\rm Fe\,I}$ straight from the equation of state of Lecture 2, carried in the reference data; nothing here is re-derived.

We pull the atmospheric structure out of the reference bundle and pick the layer nearest $\tau = 2/3$ — the "surface" the continuum emerges from — as the reference depth for the rest of the lecture. (Note: `KEV` below is $k_B$ expressed in eV K$^{-1}$, so $\chi_\ell/(k_B T)$ comes out dimensionless when $\chi_\ell$ is in eV — it is *not* keV.)""")

code(r'''# Atmospheric structure from the Lecture-2 equation of state (CGS), via the reference bundle.
T   = REF["T"]; tk = REF["tk"]; n_e = REF["xne"]; rho = REF["rho"]
n_FeI = REF["n_FeI"]; U_FeI = REF["U_FeI"]; tau = REF["tau"]
KEV = 1.0/11604.5                                          # Boltzmann k_B in eV per kelvin (NOT keV)
jp = np.argmin(np.abs(tau - 2/3))                          # index of the photosphere layer (tau ~ 2/3)
print(f"at the photosphere (T={T[jp]:.0f} K): n(Fe I) = {n_FeI[jp]:.3e} cm^-3, U(Fe I) = {U_FeI[jp]:.1f}")''')

# ── broadening + Voigt ──────────────────────────────────────────────────
md(r"""## The shape: Doppler core and Lorentzian wings

The profile $\phi(\nu)$ has two origins. **Thermal motion** Doppler-shifts each atom's absorption; averaged over a Maxwellian (plus a microturbulent velocity $\xi$ that represents unresolved small-scale flows), this gives a **Gaussian** core of $1/e$ half-width

$$
\Delta\nu_D = \frac{\nu_0}{c}\sqrt{\frac{2kT}{m} + \xi^2}.
$$

Independently, the finite lifetime of the atomic levels and perturbations from passing particles broaden the line into a **Lorentzian**, with a damping rate $\gamma = \gamma_{\rm rad} + \gamma_{\rm Stark} + \gamma_{\rm vdW}$ (in $\mathrm{s^{-1}}$): **natural** broadening (the radiative lifetime, always present), **Stark** broadening (collisions with charged particles, leading dependence $\propto n_e$), and **van der Waals** broadening (collisions with neutral hydrogen, leading dependence $\propto n_{\rm H}$, dominant in cool dwarfs). Here we emphasise the dominant density dependence; real line lists carry transition-specific broadening constants and temperature dependences, introduced in Lecture 5. Convolving the Gaussian core with the Lorentzian wings gives the **Voigt profile**

$$
\phi(\nu) = \frac{1}{\sqrt{\pi}\,\Delta\nu_D}\,H(a, v),\qquad
v = \frac{\nu - \nu_0}{\Delta\nu_D},\qquad
a = \frac{\gamma}{4\pi\,\Delta\nu_D},
$$

where $v$ measures the distance from line centre in Doppler widths and the **damping parameter** $a$ is the ratio of Lorentzian to Doppler width. (The $4\pi$ is not arbitrary: $\gamma/4\pi$ is exactly the Lorentzian half-width at half-maximum in ordinary frequency space, so $a$ is that half-width measured in Doppler widths.) The dimensionless **Voigt function** $H(a,v)$ is the heart of every line.

![The Voigt profile is a thermal Doppler (Gaussian) core convolved with pressure- and natural-broadening (Lorentzian) wings.](resources/figures/s4_voigt.png)""")

md(r"""### From the exact Voigt to Kurucz's compatibility kernel

$H(a,v)$ is, mathematically, the real part of the **Faddeeva function** $w(z) = e^{-z^2}\mathrm{erfc}(-iz)$ at $z = v + ia$, i.e. $H(a,v) = \mathrm{Re}\,w(v+ia)$. Evaluating it exactly — for example with `scipy.special.wofz` — is one option, and swapping that in to compare is a good exercise. But the standard in this book is to reproduce the reference **exactly**, and the reference does not use the Faddeeva function: it uses Kurucz's fast **Harris-series approximation**,

$$
H(a,v) \approx H_0(v) + a\,H_1(v) + a^2 H_2(v) \quad\text{(near line centre)},
$$

built from three precomputed tables — $H_0(v) = e^{-v^2}$, $H_1(v)$, and $H_2(v) = (1-2v^2)e^{-v^2}$ — with a Lorentzian-wing form far from centre and an intermediate branch between. These tables are **numerical reference values for a special function**, not atomic data like wavelengths or $gf$ values: they depend only on the dimensionless coordinate $v$, never on the element. We reuse them and reimplement the routine's exact branch logic, so our Voigt matches the gold standard to machine precision. (The Harris approximation itself differs from the true Faddeeva by up to $\sim1\%$ at large damping — a property of the reference we are matching, not an error of ours.)

The next two functions are *not* meant to derive the Harris approximation; they are a **compatibility kernel** that reproduces Kurucz's branch choices and coefficients exactly, so the comparison is bit-for-bit. The numeric constants are his, to be trusted rather than memorised; the only physical inputs are the damping parameter $a$ and the reduced frequency $v$. We first define the asymptotic Lorentzian-wing branch on its own, then the dispatcher that selects a branch by the size of $a$.""")

code(r'''h0tab, h1tab, h2tab = REF["h0tab"], REF["h1tab"], REF["h2tab"]   # Harris special-function tables (math, not atomic data)

def _voigt_wing(a, v):
    """Asymptotic (Lorentzian-wing) branch of Kurucz's Voigt approximation, valid far from centre."""
    aa, vv = a*a, v*v
    u = (aa + vv)*1.4142                                     # sqrt(2)*(a^2+v^2); sets the wing scale
    val = a*0.79788/u                                        # leading Lorentzian term, 0.79788 ~ 1/(sqrt(2)*sqrt(pi))*... (Kurucz const)
    if a <= 100.0:                                           # higher-order correction (skipped at huge damping)
        aau, vvu, uu = aa/u, vv/u, u*u
        val = ((((aau - 10.0*vvu)*aau*3.0 + 15.0*vvu*vvu) + 3.0*vv - aa)/uu + 1.0)*val
    return val''')

md(r"""The dispatcher below first maps each $|v|$ to the nearest table index (the tables are sampled every $0.005$ in $v$, hence the factor $200$), then picks one of three regimes by the damping $a$: a second-order Harris series for weak damping, the pure asymptotic wing for strong damping, and a blended branch in between that switches per point at $a + |v| = 3.2$.""")

code(r'''def voigt_H(a, v):
    """Voigt H(a,v): Kurucz's Harris-table routine, reproduced exactly (the gold-standard form)."""
    v = np.atleast_1d(np.asarray(v, float)); av = np.abs(v)
    iv = np.clip((av*200.0 + 0.5).astype(int), 0, h0tab.size-1)   # nearest table index (200 = 1/0.005 grid step)
    H0, H1, H2 = h0tab[iv], h1tab[iv], h2tab[iv]             # look up the three Harris coefficients
    out = np.empty_like(v)
    if a < 0.2:                                              # weak damping: 2nd-order Harris series
        far = av > 10.0                                      # very far out, fall back to a bare Lorentzian wing
        out[far]  = 0.5642*a/(v[far]*v[far])
        out[~far] = (H2[~far]*a + H1[~far])*a + H0[~far]     # H0 + a*H1 + a^2*H2 (Horner form)
    elif a > 1.4:                                            # strong damping: pure asymptotic wing everywhere
        out[:] = _voigt_wing(a, v)
    else:                                                    # intermediate: split per point at a+|v|=3.2
        asy = (a + av) > 3.2                                 # points far enough out use the wing branch
        out[asy] = _voigt_wing(a, v[asy])
        m = ~asy; vv = v[m]*v[m]; h0 = H0[m]; h1t = H1[m]; h2t = H2[m]   # remaining points: blended polynomial
        h1 = h1t + h0*1.12838                                # 1.12838 ~ 2/sqrt(pi): Kurucz recurrence constants
        h2 = h2t + h1*1.12838 - h0
        h3 = (1.0 - h2t)*0.37613 - h1*0.66667*vv + h2*1.12838
        h4 = (3.0*h3 - h1)*0.37613 + h0*0.66667*vv*vv
        polyA = (((h4*a + h3)*a + h2)*a + h1)*a + h0         # quartic in a, coefficients h0..h4
        polyB = ((-0.122727278*a + 0.532770573)*a - 0.96284325)*a + 0.979895032   # cubic correction in a
        out[m] = polyA*polyB
    return out''')

md(r"""With the kernel in hand, we evaluate $H(a,v)$ on the reference grid of $(a, v)$ pairs, confirm it matches pykurucz to machine precision, and plot the family of profiles for several damping values.""")

code(r'''v_ref, a_ref, H_ref = REF["v"], REF["a"], REF["H"]
H_mine = np.array([voigt_H(aa, v_ref) for aa in a_ref])     # one row per damping value
compare("Voigt H(a,v)", H_mine, H_ref)

for k, aa in enumerate(a_ref):
    plt.plot(v_ref, H_mine[k], label=f"a = {aa:g}")
plt.yscale("log"); plt.ylim(1e-4, 1.2)
plt.xlabel(r"$v = (\nu-\nu_0)/\Delta\nu_D$"); plt.ylabel(r"Voigt function $H(a,v)$")
plt.title("Doppler core, Lorentzian wings: the Voigt function"); plt.legend(); plt.tight_layout(); plt.show()''')

md(r"""The Gaussian core (small $v$) is nearly independent of $a$; the wings (large $v$) are pure Lorentzian and scale as $a/(\sqrt{\pi}v^2)$ — the larger the damping, the heavier the wings. Strong lines with broad damping wings are how we read surface gravity and pressure from a spectrum.""")

# ── assemble one line ───────────────────────────────────────────────────
md(r"""## Assembling one iron line

Now put it together for a representative neutral-iron line in our window: rest wavelength $\lambda_0 = 500.5\ \mathrm{nm}$, $\log gf = -1.0$, lower-level excitation $\chi_\ell = 3.3\ \mathrm{eV}$. We build its opacity profile at the photosphere, with a microturbulence $\xi = 2\ \mathrm{km\,s^{-1}}$ and a damping rate dominated by radiation and van der Waals collisions. The result is the line absorption coefficient per gram, $\alpha_\nu/\rho$, which we overlay on the continuous opacity of Lecture 3 — the spike that carves the line.

We first fix the line's identity (wavelength, strength, excitation) and the absorber's mass and microturbulence.""")

code(r'''# Representative neutral-iron line (demonstration values).
lam0_nm = 500.5; loggf = -1.0; chi_l = 3.3                 # rest wavelength [nm], log(gf), excitation [eV]
m_Fe = 55.845 * AMU                                        # iron atom mass [g]
xi = 2.0e5                                                  # microturbulent velocity [cm/s] = 2 km/s
nu0 = C / (lam0_nm * 1e-7)                                  # line-centre frequency [Hz]''')

md(r"""Next the two profile shape parameters: the Doppler width $\Delta\nu_D$ (thermal motion plus microturbulence) and the damping parameter $a$ from the total broadening rate $\gamma$.

A caution on the $\gamma$ values below: the natural rate is a plausible radiative lifetime, but the Stark ($\propto n_e$) and van der Waals ($\propto n_{\rm H}$) coefficients here are deliberately simple order-of-magnitude **placeholders** chosen to give a realistic-looking wing for this one demonstration line. Do not memorise `1e-7 * n_H` as the van der Waals law: real synthesis uses transition-specific constants and proper temperature scaling (e.g. the ABO theory of Anstee & O'Mara), which Lecture 5 supplies from the line list.""")

code(r'''# Doppler width and damping at the photosphere.
dnu_D = (nu0 / C) * np.sqrt(2*K*T[jp]/m_Fe + xi**2)        # Gaussian 1/e half-width [Hz]
gamma_rad = 2.0e8                                           # natural (radiative) broadening [s^-1]
gamma_vdw = 1.0e-7 * REF["nHI"][jp]                         # van der Waals, ~ n_H [s^-1]  (toy coefficient)
gamma_stark = 1.0e-8 * n_e[jp]                              # Stark, ~ n_e [s^-1]          (toy coefficient)
gamma = gamma_rad + gamma_vdw + gamma_stark                # total damping rate [s^-1]
a_damp = gamma / (4*np.pi*dnu_D)                            # damping parameter (Lorentzian / Doppler width)
dlam_mA = dnu_D * (lam0_nm*1e-7)**2 / C * 1e8 * 1e3        # Doppler width converted to milli-angstrom
print(f"Doppler width = {dnu_D:.3e} Hz ({dlam_mA:.0f} mA);  damping a = {a_damp:.3f}")''')

md(r"""Finally we evaluate the full absorption coefficient on a wavelength grid: the classical constant times $10^{\log gf}$ (strength), the lower-level population per sublevel (Boltzmann), the normalised Voigt profile (shape), and the stimulated-emission factor. Dividing by the mass density gives the opacity per gram, directly comparable with the Lecture-3 continuum.""")

code(r'''# Opacity profile across +-0.04 nm around line centre.
lam = np.linspace(lam0_nm-0.04, lam0_nm+0.04, 800)
nu = C / (lam*1e-7)
v = (nu - nu0) / dnu_D                                      # reduced frequency in Doppler widths
phi = voigt_H(a_damp, v) / (np.sqrt(np.pi) * dnu_D)        # normalised line profile [s]
stim = 1.0 - np.exp(-H_C*nu/(K*T[jp]))                     # stimulated-emission factor (Lecture 1)
nl_over_gl = (n_FeI[jp]/U_FeI[jp]) * np.exp(-chi_l/(KEV*T[jp]))   # Boltzmann population per sublevel (Lecture 2)
alpha_line = CLASSICAL * 10**loggf * nl_over_gl * phi * stim    # line absorption coefficient [cm^-1]
kappa_line = alpha_line / rho[jp]                              # opacity per gram [cm^2/g]
print(f"line-centre opacity / continuum = {kappa_line.max()/0.027:.1f}  (continuum ~0.027 cm^2/g)")''')

md(r"""Plotting the line opacity stacked on a representative H$^-$ continuum ($\sim0.027\ \mathrm{cm^2/g}$) shows the spike that carves the absorption feature.""")

code(r'''plt.plot(lam, kappa_line + 0.027, color="C3", label="line + continuum")
plt.axhline(0.027, ls="--", color="0.5", lw=1, label="continuum (H$^-$)")
plt.yscale("log"); plt.xlabel("wavelength [nm]"); plt.ylabel(r"$\kappa$  [cm$^2$/g]")
plt.title(r"Opacity profile of one Fe I line at the photosphere ($\lambda_0=500.5$ nm)")
plt.legend(); plt.tight_layout(); plt.show()''')

md(r"""The line raises the opacity more than a hundredfold at its centre, falling through a Gaussian core into Lorentzian wings that merge back into the continuum a fraction of an ångström away. At line centre the $\tau=2/3$ surface is pushed up into cool layers and the emergent flux drops — that conversion of opacity into a flux profile is the radiative transfer of Lecture 7. For now, note the levers we have exposed: $\log gf$ and the Boltzmann factor scale the whole profile up or down, while the damping $a$ controls how much of the line lives in the wings. Vary $\log gf$ (or the abundance hidden inside $n(\mathrm{Fe\,I})$) and you trace the **curve of growth** — the relation between a line's strength and its equivalent width that turns a measured depth into an abundance.""")

# ── close ───────────────────────────────────────────────────────────────
md(r"""## Synthesis: what you built and where it goes

You built a single spectral line from its parts. Its **strength** is the classical constant times the oscillator strength `log gf` times the lower-level population — and that population comes from the **ionization** (Saha) and **excitation** (Boltzmann, through $\chi_\ell$) of Lecture 2. Its **shape** is the **Voigt profile** — mathematically the real part of the Faddeeva function, reproduced here through Kurucz's Harris-table approximation — convolving a thermal-plus-turbulent Doppler core with natural, Stark, and van der Waals wings. You overlaid the result on the H$^-$ continuum and saw the opacity spike that becomes an absorption line.

One line is a demonstration; a real spectrum is a forest. Lecture 5 reads the Kurucz line list — over a million atomic transitions — assigns each its $\log gf$, excitation potential, and damping constants, and sums their Voigt profiles onto a wavelength grid to build the total line opacity at every depth, ready for the radiative transfer that turns it into the spectrum.""")

md(r"""## Summary

- The line absorption coefficient is $\alpha_\nu = (\pi e^2/m_e c)\,f_{\ell u}\,n_\ell\,\phi(\nu)\,(1-e^{-h\nu/kT})$, with $\pi e^2/m_e c = 0.02654\ \mathrm{cm^2\,Hz}$.
- Line lists give **$\log gf$**; the opacity uses $gf\cdot(n_\ell/g_\ell)$, and $n_\ell/g_\ell = (n_{\rm ion}/U)\,e^{-\chi_\ell/kT}$ from Boltzmann.
- The profile is a **Voigt** function $\phi = H(a,v)/(\sqrt{\pi}\Delta\nu_D)$: Gaussian Doppler core (thermal + microturbulence) and Lorentzian wings from natural + Stark + van der Waals broadening, with damping $a=\gamma/4\pi\Delta\nu_D$.
- $H(a,v)$ is Kurucz's Harris-table approximation to $\mathrm{Re}\,w(v+ia)$, reproduced **exactly** (machine precision) by reusing its tables and branch logic; the true Faddeeva is a one-line swap-in for comparison.
- $\log gf$, the excitation potential, and the damping are the levers connecting a line's depth to temperature, gravity, and abundance.""")

md(r"""## Practice exercises

**1. Excitation potential and depth.** Recompute the line's central opacity for $\chi_\ell = 0$ and $\chi_\ell = 5\ \mathrm{eV}$ at three depths. Which line is relatively stronger in the cool upper photosphere, and why does high-excitation make a line a deeper-forming, more temperature-sensitive probe?

**2. The damping wings.** Increase $\gamma_{\rm vdW}$ by factors of $3$ and $10$ (mimicking higher gravity) and plot the profiles. How does the damping parameter $a$ change, and where in the profile does the extra opacity appear? This is the surface-gravity diagnostic.

**3. Equivalent width.** Numerically integrate the line depth $1 - e^{-\tau_{\rm line}}$ across the line for a range of $\log gf$ from $-4$ to $0$, and plot equivalent width against $\log gf$. For this toy curve of growth, treat the line optical depth as $\tau_{\rm line} = \kappa_{\rm line}/\kappa_{\rm cont}$ with a fixed normalisation (a single constant of order unity); this is not a formal equivalent width, so what matters is the *shape* of the three regimes — the linear, flat (saturated), and damping parts of the curve of growth.""")

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
