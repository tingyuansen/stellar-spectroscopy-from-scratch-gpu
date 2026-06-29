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
def code(s): cells.append(new_code_cell(s.strip("\n")))

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

The continuum of Lecture 3 is the smooth canvas; a spectral **line** comes from a bound-bound transition, where a bound electron jumps between two specific energy levels. The transition has a single rest frequency, but broadening spreads the extra opacity over a band of nearby frequencies. In an LTE photosphere with temperature decreasing outward, that added opacity shifts the effective formation height upward to cooler, dimmer layers, so the feature appears in absorption. Two questions define a line. *How strong is it* — how much opacity does it add at line centre? And *what shape* does it have — how is that opacity spread over wavelength?

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

    # Guard against divide-by-zero where the reference value is exactly zero.
    denom = np.where(ref != 0.0, np.abs(ref), 1.0)

    rel = float(np.max(np.abs(ours - ref) / denom))
    tag = "exact" if rel < 1e-12 else ("agree" if rel < tol else "CHECK")
    print(f"{name:30s}  max|rel diff| = {rel:.2e}   [{tag}]")
    return rel

# Physical constants in CGS units.
# H_C, C, K are Planck's h, the speed of light c, and Boltzmann's k, in that order.
H_C, C, K = 6.62607015e-27, 2.99792458e10, 1.380649e-16
AMU = 1.66053907e-24    # [g]

print("reference loaded:", ", ".join(REF.files))''')

# ── line strength ───────────────────────────────────────────────────────
md(r"""## The strength of a line

The monochromatic absorption coefficient of a bound-bound transition, per unit volume, is

$$
\alpha_\nu = \frac{\pi e^2}{m_e c}\, f_{\ell u}\, n_\ell\, \phi(\nu)\,\big(1 - e^{-h\nu/kT}\big)\quad[\mathrm{cm^{-1}}],
$$

a product of four pieces. In LTE this is the net bound-bound absorption coefficient, the upward absorption after subtracting stimulated emission; outside LTE the final factor is replaced by the corresponding upper/lower population ratio. The classical constant $\pi e^2/m_e c = 0.02654\ \mathrm{cm^2\,Hz}$ is the cross-section a classical oscillator would present, integrated over frequency. The **oscillator strength** $f_{\ell u}$ — a dimensionless quantum-mechanical number, usually of order unity or smaller but not strictly bounded by 1 — corrects that classical value for the real transition. The **lower-level population** $n_\ell$ counts how many absorbers are available. And $\phi(\nu)$ is the line profile, normalised to $\int\phi\,d\nu = 1$, which spreads the opacity over frequency. The last factor is the same stimulated-emission correction $(1 - e^{-h\nu/kT})$ we met in Lecture 1; nothing new here.

Line lists do not tabulate $f$ directly; they give **$\log gf$**, the base-ten logarithm of the oscillator strength times the lower level's statistical weight $g_\ell$. The combination is what appears in the opacity, because $f_{\ell u}\,n_\ell = gf \cdot (n_\ell/g_\ell)$, and $n_\ell/g_\ell$ is exactly what the Boltzmann factor gives without needing $g_\ell$ separately. Thus the tabulated $gf$ is not a new physical factor; it is a bookkeeping device that pairs naturally with the Boltzmann population *per sublevel*, $n_\ell/g_\ell$. So $\log gf$ is the tabulated radiative-strength factor — a strong line might have $\log gf \approx 0$, a weak one $\log gf \approx -5$ — while the realized line opacity then also depends on the population, abundance, and atmospheric conditions.

The code cell below evaluates only the classical constant; the remaining factors are assembled later once we have the population and profile.""")

code(r'''# Classical line constant pi e^2 / (m_e c), from the electron charge and mass in CGS.
CLASSICAL = np.pi * (4.803204e-10)**2 / (9.1093837e-28 * C)    # [cm^2 Hz]

print(f"classical line constant  pi e^2 / (m_e c) = {CLASSICAL:.5e} cm^2 Hz")''')

md(r"""## Populating the lower level

How many atoms sit in the lower level of our transition? Two factors set it, both from Lecture 2. First, what fraction of the element is in the right ionization stage — here neutral iron, Fe I. Second, within that ion, the **Boltzmann** factor places a fraction in the lower level $\ell$ at excitation energy $\chi_\ell$ above the ground state:

$$
\frac{n_\ell}{g_\ell} = \frac{n(\mathrm{Fe\,I})}{U_{\rm Fe\,I}(T)}\,e^{-\chi_\ell/kT}.
$$

The **excitation potential** $\chi_\ell$ is decisive: a ground-state line ($\chi_\ell = 0$) carries no excitation penalty, though its availability still depends on the ionization balance, while a line from a level several eV up is populated only in the hotter, deeper layers. All else equal, higher-excitation lines are weighted toward hotter, deeper layers — the lever that lets spectroscopy measure temperature. We take the number density $n(\mathrm{Fe\,I})$ and the partition function $U_{\rm Fe\,I}$ straight from the equation of state of Lecture 2, carried in the reference data; nothing here is re-derived.

We pull the atmospheric structure out of the reference bundle and pick the layer nearest $\tau = 2/3$ — the "surface" the continuum emerges from — as the reference depth for the rest of the lecture. (Note: `KEV` below is $k_B$ expressed in eV K$^{-1}$, so $\chi_\ell/(k_B T)$ comes out dimensionless when $\chi_\ell$ is in eV — it is *not* keV.)""")

code(r'''# Atmospheric structure from the Lecture-2 equation of state (CGS), via the reference bundle.
T   = REF["T"]; tk = REF["tk"]; n_e = REF["xne"]; rho = REF["rho"]
n_FeI = REF["n_FeI"]; U_FeI = REF["U_FeI"]; tau = REF["tau"]

# Boltzmann k_B in eV per kelvin (NOT keV), so chi/(KEV*T) is dimensionless for chi in eV.
KEV = 1.0/11604.5

# Photosphere layer: the one nearest tau ~ 2/3, the depth the continuum emerges from.
jp = np.argmin(np.abs(tau - 2/3))

print(f"at the photosphere (T={T[jp]:.0f} K): n(Fe I) = {n_FeI[jp]:.3e} cm^-3, U(Fe I) = {U_FeI[jp]:.1f}")''')

# ── broadening + Voigt ──────────────────────────────────────────────────
md(r"""## The shape: Doppler core and Lorentzian wings

The profile $\phi(\nu)$ has two origins. **Thermal motion** Doppler-shifts each atom's absorption; averaged over a Maxwellian (plus a microturbulent velocity $\xi$ that represents unresolved small-scale flows), this gives a **Gaussian** core of $1/e$ half-width

$$
\Delta\nu_D = \frac{\nu_0}{c}\sqrt{\frac{2kT}{m} + \xi^2}.
$$

Independently, the finite lifetime of the atomic levels and perturbations from passing particles broaden the line into a **Lorentzian**, with a damping rate $\gamma = \gamma_{\rm rad} + \gamma_{\rm Stark} + \gamma_{\rm vdW}$ (in $\mathrm{s^{-1}}$): **natural** broadening (set by the inverse of the finite radiative lifetimes of the transition's levels, i.e. the sum of the spontaneous-decay rates out of them, always present), **Stark** broadening (for the metal-line treatment used here, the electron-impact contribution scales roughly $\propto n_e$; hydrogen and some strong lines need more specialized Stark theory), and **van der Waals** broadening (collisions with neutral perturbers, leading dependence $\propto n_{\rm H}$, usually dominated by neutral H in cool dwarf photospheres). Here we emphasise the dominant density dependence; real line lists carry transition-specific broadening constants and temperature dependences, introduced in Lecture 5. Convolving the full Gaussian Doppler profile with the full Lorentzian damping profile gives the **Voigt profile**

$$
\phi(\nu) = \frac{1}{\sqrt{\pi}\,\Delta\nu_D}\,H(a, v),\qquad
v = \frac{\nu - \nu_0}{\Delta\nu_D},\qquad
a = \frac{\gamma}{4\pi\,\Delta\nu_D},
$$

where $v$ measures the distance from line centre in Doppler widths and the **damping parameter** $a$ is the ratio of Lorentzian to Doppler width. (The $4\pi$ is not arbitrary: $\gamma/4\pi$ is exactly the Lorentzian half-width at half-maximum in ordinary frequency space, so $a$ is that half-width measured in Doppler widths.) The dimensionless **Voigt function** $H(a,v)$ is the heart of every line.

![The Voigt profile is the convolution of the full Gaussian (thermal Doppler) profile with the full Lorentzian (natural plus pressure) profile; the result has a Doppler-dominated core and Lorentzian damping wings.](resources/figures/s4_voigt.png)""")

md(r"""### From the exact Voigt to Kurucz's compatibility kernel

$H(a,v)$ is, mathematically, the real part of the **Faddeeva function** $w(z) = e^{-z^2}\mathrm{erfc}(-iz)$ at $z = v + ia$, i.e. $H(a,v) = \mathrm{Re}\,w(v+ia)$. Evaluating it exactly — for example with `scipy.special.wofz` — is one option, and swapping that in to compare is a good exercise. But the standard in this book is to reproduce the reference **exactly**, and the reference does not use the Faddeeva function: it uses Kurucz's fast **Harris-series approximation**,

$$
H(a,v) \approx H_0(v) + a\,H_1(v) + a^2 H_2(v) \quad\text{(near line centre)},
$$

built from three precomputed tables — $H_0(v) = e^{-v^2}$, $H_1(v)$, and $H_2(v) = (1-2v^2)e^{-v^2}$ — with a Lorentzian-wing form far from centre and an intermediate branch between. These tables are **numerical reference values for a special function**, not atomic data like wavelengths or $gf$ values: they depend only on the dimensionless coordinate $v$, never on the element. We reuse them and reimplement the routine's exact branch logic, so our Voigt matches the gold standard to machine precision. (The Harris approximation itself differs from the true Faddeeva by up to $\sim1\%$ at large damping — a property of the reference we are matching, not an error of ours.)

The next two functions are *not* meant to derive the Harris approximation; they are a **compatibility kernel** that reproduces Kurucz's branch choices and coefficients exactly, so the comparison agrees to roundoff precision against the reference arrays. The numeric constants are his, to be trusted rather than memorised; the only physical inputs are the damping parameter $a$ and the reduced frequency $v$. We first define the asymptotic Lorentzian-wing branch on its own, then the dispatcher that selects a branch by the size of $a$.""")

code(r'''# Harris special-function tables (numerical math, not atomic data).
h0tab, h1tab, h2tab = REF["h0tab"], REF["h1tab"], REF["h2tab"]

def _voigt_wing(a, v):
    """Asymptotic (Lorentzian-wing) branch of Kurucz's Voigt approximation, valid far from centre."""
    aa, vv = a*a, v*v

    # sqrt(2)*(a^2+v^2): the scale that sets the wing.
    u = (aa + vv)*1.4142

    # Leading Lorentzian term; 0.79788 ~ 1/(sqrt(2)*sqrt(pi))*... is one of Kurucz's constants.
    val = a*0.79788/u

    # Higher-order correction, skipped at huge damping where it is negligible.
    if a <= 100.0:
        aau, vvu, uu = aa/u, vv/u, u*u
        val = ((((aau - 10.0*vvu)*aau*3.0 + 15.0*vvu*vvu) + 3.0*vv - aa)/uu + 1.0)*val
    return val''')

md(r"""The dispatcher below first maps each $|v|$ to the nearest table index (the tables are sampled every $0.005$ in $v$, hence the factor $200$), then picks one of three regimes by the damping $a$: a second-order Harris series for weak damping, the pure asymptotic wing for strong damping, and a blended branch in between that switches per point at $a + |v| = 3.2$.""")

code(r'''def voigt_H(a, v):
    """Voigt H(a,v): Kurucz's Harris-table routine, reproduced exactly (the gold-standard form)."""
    v = np.atleast_1d(np.asarray(v, float)); av = np.abs(v)

    # Nearest table index for each |v| (200 = 1/0.005, the grid step), then the three Harris coefficients there.
    iv = np.clip((av*200.0 + 0.5).astype(int), 0, h0tab.size-1)
    H0, H1, H2 = h0tab[iv], h1tab[iv], h2tab[iv]
    out = np.empty_like(v)

    # Weak damping: the 2nd-order Harris series.
    if a < 0.2:

        # Very far out, fall back to a bare Lorentzian wing.
        far = av > 10.0
        out[far]  = 0.5642*a/(v[far]*v[far])

        # Near centre, the series H0 + a*H1 + a^2*H2 in Horner form.
        out[~far] = (H2[~far]*a + H1[~far])*a + H0[~far]

    # Strong damping: the pure asymptotic wing everywhere.
    elif a > 1.4:
        out[:] = _voigt_wing(a, v)

    # Intermediate: split per point, using the wing branch where a+|v| > 3.2.
    else:
        asy = (a + av) > 3.2
        out[asy] = _voigt_wing(a, v[asy])

        # Remaining (near-centre) points use a blended polynomial.
        m = ~asy; vv = v[m]*v[m]; h0 = H0[m]; h1t = H1[m]; h2t = H2[m]

        # 1.12838 ~ 2/sqrt(pi): Kurucz's recurrence constants that build the higher H-terms.
        h1 = h1t + h0*1.12838
        h2 = h2t + h1*1.12838 - h0
        h3 = (1.0 - h2t)*0.37613 - h1*0.66667*vv + h2*1.12838
        h4 = (3.0*h3 - h1)*0.37613 + h0*0.66667*vv*vv

        # Quartic in a (coefficients h0..h4) times a cubic correction in a.
        polyA = (((h4*a + h3)*a + h2)*a + h1)*a + h0
        polyB = ((-0.122727278*a + 0.532770573)*a - 0.96284325)*a + 0.979895032
        out[m] = polyA*polyB
    return out''')

md(r"""With the kernel in hand, we evaluate $H(a,v)$ on the reference grid of $(a, v)$ pairs, confirm it matches pykurucz to machine precision, and plot the family of profiles for several damping values.""")

code(r'''v_ref, a_ref, H_ref = REF["v"], REF["a"], REF["H"]

# Verification: the same routine, evaluated on the reference v-grid, checked against pykurucz.
# One row per damping value in a_ref.
H_mine = np.array([voigt_H(aa, v_ref) for aa in a_ref])
compare("Voigt H(a,v)", H_mine, H_ref)

# Plotting only: a denser v-grid so the smooth parts of each curve render cleanly.
# (This changes nothing in the verified computation above; voigt_H is identical.)
v_plot = np.linspace(v_ref.min(), v_ref.max(), 4000)
for aa in a_ref:
    plt.plot(v_plot, voigt_H(aa, v_plot), label=f"a = {aa:g}")
plt.yscale("log"); plt.ylim(1e-4, 1.2)
plt.xlabel(r"reduced frequency  $v = (\nu-\nu_0)/\Delta\nu_D$  [dimensionless]")
plt.ylabel(r"Voigt function  $H(a,v)$  [dimensionless]")
plt.title("The Voigt function: Doppler-dominated core, Lorentzian wings")
plt.legend(); plt.tight_layout(); plt.show()''')

md(r"""For small-to-moderate damping the central core (small $v$) is nearly Gaussian and insensitive to $a$; far enough from line centre the profile approaches Lorentzian wings (large $v$) that scale as $a/(\sqrt{\pi}v^2)$ — the larger the damping, the heavier the wings. Pressure-broadened wings of suitable strong lines are key diagnostics of surface gravity and gas pressure.

The faint kinks in the $a=0.5$ and $a=1$ curves around $v \approx 2$–$3$ are not numerical noise: they are the seams where Kurucz's piecewise Harris approximation switches between its near-centre and asymptotic-wing branches (the per-point cut at $a+|v|=3.2$). The true Voigt function is perfectly smooth there; we render the kinks faithfully because the whole point of this kernel is to reproduce the reference exactly, seams and all.""")

# ── assemble one line ───────────────────────────────────────────────────
md(r"""## Assembling one iron line

Now put it together for a representative neutral-iron line in our window: rest wavelength $\lambda_0 = 500.5\ \mathrm{nm}$, $\log gf = -1.0$, lower-level excitation $\chi_\ell = 3.3\ \mathrm{eV}$. We build its opacity profile at the photosphere, with a microturbulence $\xi = 2\ \mathrm{km\,s^{-1}}$ and a damping rate dominated by radiation and van der Waals collisions. The result is the line absorption coefficient per gram, $\alpha_\nu/\rho$, which we overlay on the continuous opacity of Lecture 3 — the spike that carves the line.

We first fix the line's identity (wavelength, strength, excitation) and the absorber's mass and microturbulence.""")

code(r'''# Representative neutral-iron line (demonstration values).
# Three identity numbers: rest wavelength [nm], log(gf), and lower-level excitation [eV].
lam0_nm = 500.5; loggf = -1.0; chi_l = 3.3

# Iron atom mass.
m_Fe = 55.845 * AMU    # [g]

# Microturbulent velocity, 2 km/s.
xi = 2.0e5    # [cm/s]

# Line-centre frequency.
nu0 = C / (lam0_nm * 1e-7)    # [Hz]''')

md(r"""Next the two profile shape parameters: the Doppler width $\Delta\nu_D$ (thermal motion plus microturbulence) and the damping parameter $a$ from the total broadening rate $\gamma$.

A caution on the $\gamma$ values below: the natural rate is a plausible radiative lifetime, but the Stark ($\propto n_e$) and van der Waals ($\propto n_{\rm H}$) coefficients here are deliberately simple order-of-magnitude **placeholders** chosen to give a realistic-looking wing for this one demonstration line. Do not memorise `1e-7 * n_H` as the van der Waals law: real synthesis uses transition-specific constants and proper temperature scaling (e.g. the ABO theory of Anstee & O'Mara), which Lecture 5 supplies from the line list.""")

code(r'''# Doppler width at the photosphere: Gaussian 1/e half-width from thermal motion plus microturbulence.
dnu_D = (nu0 / C) * np.sqrt(2*K*T[jp]/m_Fe + xi**2)    # [Hz]

# The three broadening channels (all as rates), summed into the total damping rate.
# The Stark and van der Waals coefficients are toy placeholders (see the caution above).
gamma_rad = 2.0e8                       # natural / radiative [s^-1]
gamma_vdw = 1.0e-7 * REF["nHI"][jp]     # van der Waals, ~ n_H [s^-1]
gamma_stark = 1.0e-8 * n_e[jp]          # Stark, ~ n_e [s^-1]
gamma = gamma_rad + gamma_vdw + gamma_stark    # total [s^-1]

# Damping parameter: Lorentzian half-width measured in Doppler widths.
a_damp = gamma / (4*np.pi*dnu_D)

# Doppler width converted to milli-angstrom, for an intuitive feel of its size.
dlam_mA = dnu_D * (lam0_nm*1e-7)**2 / C * 1e8 * 1e3    # [mA]

print(f"Doppler width = {dnu_D:.3e} Hz ({dlam_mA:.0f} mA);  damping a = {a_damp:.3f}")''')

md(r"""Finally we evaluate the full absorption coefficient on a wavelength grid: the classical constant times $10^{\log gf}$ (strength), the lower-level population per sublevel (Boltzmann), the normalised Voigt profile (shape), and the stimulated-emission factor. Dividing by the mass density gives the opacity per gram, directly comparable with the Lecture-3 continuum.

We are plotting the per-frequency opacity $\alpha_\nu$ at the frequency corresponding to each wavelength; the profile normalisation stays in frequency units, so no per-wavelength Jacobian is applied.""")

code(r'''# Wavelength grid across +-0.04 nm around line centre, and the matching frequencies.
lam = np.linspace(lam0_nm-0.04, lam0_nm+0.04, 800)
nu = C / (lam*1e-7)

# Reduced frequency: distance from line centre in Doppler widths.
v = (nu - nu0) / dnu_D

# Normalised line profile (shape) from the Voigt function.
phi = voigt_H(a_damp, v) / (np.sqrt(np.pi) * dnu_D)    # [s]

# Stimulated-emission factor (Lecture 1).
stim = 1.0 - np.exp(-H_C*nu/(K*T[jp]))

# Boltzmann population per sublevel (Lecture 2).
nl_over_gl = (n_FeI[jp]/U_FeI[jp]) * np.exp(-chi_l/(KEV*T[jp]))

# Assemble the four factors into the line absorption coefficient.
alpha_line = CLASSICAL * 10**loggf * nl_over_gl * phi * stim    # [cm^-1]

# Divide by mass density for the opacity per gram.
kappa_line = alpha_line / rho[jp]    # [cm^2/g]

print(f"line-centre opacity / continuum = {kappa_line.max()/0.027:.1f}  (continuum ~0.027 cm^2/g)")''')

md(r"""Plotting the line opacity stacked on a representative H$^-$ continuum ($\sim0.027\ \mathrm{cm^2/g}$) shows the spike that carves the absorption feature.""")

code(r'''plt.plot(lam, kappa_line + 0.027, color="C3", label="line + continuum")
plt.axhline(0.027, ls="--", color="0.5", lw=1, label="continuum (H$^-$)")
plt.yscale("log"); plt.xlabel("wavelength [nm]"); plt.ylabel(r"$\kappa$  [cm$^2$/g]")
plt.title(r"Opacity profile of one Fe I line at the photosphere ($\lambda_0=500.5$ nm)")
plt.legend(); plt.tight_layout(); plt.show()''')

md(r"""The line raises the opacity more than a hundredfold at its centre, falling through a Gaussian core into Lorentzian wings that merge back into the continuum a fraction of an ångström away. At line centre the $\tau=2/3$ surface is pushed up into cool layers and the emergent flux drops — that conversion of opacity into a flux profile is the radiative transfer of Lecture 7. For now, note the levers we have exposed: $\log gf$ and the Boltzmann factor scale the whole profile up or down, while the damping $a$ controls how much of the line lives in the wings. Vary $\log gf$ (or the abundance hidden inside $n(\mathrm{Fe\,I})$) and you trace the **curve of growth** — the relation between a line's strength and its equivalent width that turns a measured equivalent width, or fitted line profile, into an abundance.""")

# ── close ───────────────────────────────────────────────────────────────
md(r"""## Synthesis: what you built and where it goes

You built a single spectral line from its parts. Its **strength** is the classical constant times the oscillator strength `log gf` times the lower-level population — and that population comes from the **ionization** (Saha) and **excitation** (Boltzmann, through $\chi_\ell$) of Lecture 2. Its **shape** is the **Voigt profile** — mathematically the real part of the Faddeeva function, reproduced here through Kurucz's Harris-table approximation — the convolution of the full thermal-plus-turbulent Gaussian Doppler profile with the full Lorentzian damping profile from natural, Stark, and van der Waals broadening. You overlaid the result on the H$^-$ continuum and saw the opacity spike that becomes an absorption line.

One line is a demonstration; a real spectrum is a forest. Lecture 5 reads the Kurucz line list — over a million atomic transitions — assigns each its $\log gf$, excitation potential, and damping constants, and sums their Voigt profiles onto a wavelength grid to build the total line opacity at every depth, ready for the radiative transfer that turns it into the spectrum.""")

md(r"""## Summary

- The line absorption coefficient is $\alpha_\nu = (\pi e^2/m_e c)\,f_{\ell u}\,n_\ell\,\phi(\nu)\,(1-e^{-h\nu/kT})$, with $\pi e^2/m_e c = 0.02654\ \mathrm{cm^2\,Hz}$.
- Line lists give **$\log gf$**; the opacity uses $gf\cdot(n_\ell/g_\ell)$, and $n_\ell/g_\ell = (n_{\rm ion}/U)\,e^{-\chi_\ell/kT}$ from Boltzmann.
- The profile is a **Voigt** function $\phi = H(a,v)/(\sqrt{\pi}\Delta\nu_D)$, the convolution of the full Gaussian Doppler profile (thermal + microturbulence) with the full Lorentzian damping profile (natural + Stark + van der Waals), giving a Doppler-dominated core and Lorentzian wings, with damping $a=\gamma/4\pi\Delta\nu_D$.
- $H(a,v)$ is Kurucz's Harris-table approximation to $\mathrm{Re}\,w(v+ia)$, reproduced to machine precision against the reference arrays by reusing its tables and branch logic; the true Faddeeva is a one-line swap-in for comparison.
- $\log gf$, the excitation potential, and the damping are the levers connecting a line's depth to temperature, gravity, and abundance.""")

md(r"""## Practice exercises

**1. Excitation potential and depth.** Recompute the line's central opacity for $\chi_\ell = 0$ and $\chi_\ell = 5\ \mathrm{eV}$ at three depths. Which line is relatively stronger in the cool upper photosphere, and why does high-excitation make a line a deeper-forming, more temperature-sensitive probe?

**2. The damping wings.** Increase $\gamma_{\rm vdW}$ by factors of $3$ and $10$ (mimicking higher gravity) and plot the profiles. How does the damping parameter $a$ change, and where in the profile does the extra opacity appear? This is the surface-gravity diagnostic.

**3. Toy equivalent-width proxy.** Numerically integrate the line depth $1 - e^{-\tau_{\rm line}}$ over wavelength across the line for a range of $\log gf$ from $-4$ to $0$, and plot this proxy against $\log gf$. Treat the line optical depth as $\tau_{\rm line} = \kappa_{\rm line}/\kappa_{\rm cont}$ with a fixed normalisation (a single constant of order unity); this is not a formal equivalent width from radiative transfer, so what matters is the *shape* of the three regimes — the linear, flat (saturated), and damping parts of the curve of growth.""")

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
