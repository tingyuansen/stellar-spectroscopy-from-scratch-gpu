#!/usr/bin/env python
"""Lecture 7 — Radiative Transfer & the Emergent Spectrum. The transfer equation,
the LTE source function, optical depth, the formal solution F = 2*pi*int S E2(tau) dtau,
and the assembled solar spectrum. Checked against reference/L6.npz. No pykurucz import.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture7.ipynb"
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# Lecture 7 — Radiative Transfer & the Emergent Spectrum

*Stellar Spectroscopy from Scratch — rebuilding the physics of ATLAS and SYNTHE from first principles*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*Every result in this book is checked against reference values computed with [**pykurucz**](https://arxiv.org/abs/2603.11693) — a pure-Python implementation of Kurucz's ATLAS12 and SYNTHE — shipped beside the lectures as small data files, so the notebooks need only NumPy to run.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Write the **plane-parallel transfer equation** and its **formal solution**, and identify the LTE source function.
- Turn an opacity into an **optical-depth scale** by integrating over column mass.
- Compute the **emergent flux** with the exponential-integral formal solution $F_\lambda = 2\pi\int S_\lambda E_2(\tau)\,d\tau$, and recover the **Eddington–Barbier** relation.
- Assemble the full **solar spectrum** from $500$ to $510\ \mathrm{nm}$ and reproduce the reference to better than a part in a thousand.
- Explain where, and why, the deepest line cores deviate — the signature of scattering.""")

md(r"""## Introduction

We have everything the photons need. Lecture 1 gave the temperature structure; Lecture 2 the populations; Lecture 3 the continuous opacity; Lectures 4–5 the line opacity. What remains is to let the light out: to solve the **radiative transfer equation** for the intensity that emerges from the top of the atmosphere, wavelength by wavelength. That emergent flux *is* the spectrum.

The physics is a single first-order equation, and in LTE it has a closed-form **formal solution** — the emergent flux is a weighted average of the source function over depth, with the weighting set by how the optical depth builds up. We will write that solution, evaluate it on the opacity we assembled, and watch a forest of iron lines appear exactly where the line list said they would. This is the lecture where the whole pipeline pays off: a synthetic solar spectrum, built from $T_{\rm eff}$ and $\log g$, matching the production code to a part in a thousand.""")

code(r'''import pathlib
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5})

# Lecture 7 starts from the opacity products saved at the end of Lecture 6 (hence "L6").
REF = np.load(pathlib.Path("..") / "reference" / "L6.npz")
def compare(name, ours, ref, tol=1e-6):
    """Report how closely a from-scratch array matches the reference values."""
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    denom = np.where(ref != 0.0, np.abs(ref), 1.0)        # avoid divide-by-zero on exact zeros
    rel = float(np.max(np.abs(ours - ref) / denom))
    tag = "exact" if rel < 1e-12 else ("agree" if rel < tol else "CHECK")
    print(f"{name:30s}  max|rel diff| = {rel:.2e}   [{tag}]")
    return rel

H_C, C, K = 6.62607015e-27, 2.99792458e10, 1.380649e-16   # Planck h, speed of light c, Boltzmann k  [CGS]
wl = REF["wl"]; T = REF["T"]; rhox = REF["rhox"]; tau_ross = REF["tau"]   # grid: wavelength, T(depth), column mass, Rosseland tau
# Opacity assembled in the continuum (Lecture 3) and line (Lectures 4-6) lectures, per depth and wavelength [cm^2/g].
# cont_* are continuum processes only; total_* add all the line opacity on top of the continuum.
total_abs = REF["total_abs"].astype(float); total_scat = REF["total_scat"].astype(float)   # full opacity: absorption, scattering
cont_abs = REF["cont_abs"].astype(float);   cont_scat = REF["cont_scat"].astype(float)      # continuum-only: absorption, scattering
print(f"opacity grid: {total_abs.shape[0]} depths x {wl.size} wavelengths, {wl[0]:.1f}-{wl[-1]:.1f} nm")''')

# ── transfer equation ───────────────────────────────────────────────────
md(r"""## The transfer equation and its formal solution

Along a ray at angle $\theta$ to the surface normal ($\mu = \cos\theta$), the specific intensity obeys the **plane-parallel transfer equation**

$$
\mu\,\frac{dI_\lambda}{d\tau_\lambda} = I_\lambda - S_\lambda,
$$

where $\tau_\lambda$ is the monochromatic optical depth measured inward and $S_\lambda$ is the **source function** — emissivity divided by the extinction coefficient (absorption *plus* scattering). In **local thermodynamic equilibrium** with negligible scattering this reduces to the Planck function at the local temperature, $S_\lambda = B_\lambda(T)$, the result we have leaned on since Lecture 1.

This equation integrates exactly. For an emergent ray the direction cosine is strictly outward, $0 < \mu \le 1$, and the intensity emerging at the surface ($\tau=0$) is the source function summed along the ray, weighted by the attenuation back to the surface,

$$
I_\lambda(0, \mu) = \int_0^\infty S_\lambda(\tau_\lambda)\,e^{-\tau_\lambda/\mu}\,\frac{d\tau_\lambda}{\mu}.
$$

The **flux** is this intensity integrated over the outward hemisphere ($0 < \mu \le 1$). Carrying out the angle integral turns the exponential into the **second exponential integral** $E_2(\tau) = \int_1^\infty t^{-2}e^{-\tau t}\,dt$, giving the compact result

$$
F_\lambda = 2\pi \int_0^\infty S_\lambda(\tau_\lambda)\,E_2(\tau_\lambda)\,d\tau_\lambda .
$$

The flux is a weighted average of the source function over depth. The kernel $E_2(\tau)$ is largest at $\tau=0$ and falls off outward, so the emergent flux samples a finite range of optical depths near the surface. For a source function that varies roughly *linearly* with optical depth — the usual case — that weighted average comes out close to $\pi\,S_\lambda(\tau_\lambda \approx 2/3)$, the **Eddington–Barbier** depth previewed in Lecture 1. We see, at each wavelength, the source function around $\tau_\lambda \approx 2/3$; where a line spikes the opacity, that surface rises into cooler gas and the flux drops. (We make the $2/3$ precise below.)""")

# ── optical depth + formal solution ─────────────────────────────────────
md(r"""## From opacity to optical depth

To use the formal solution we need the optical-depth scale at every wavelength. Optical depth accumulates the extinction (absorption plus scattering) along the column, and because our opacities are per gram, the integral is over the **column mass** $\rho x$ (`RHOX`) we have carried since Lecture 1:

$$
\tau_\lambda(\rho x) = \int_0^{\rho x} \big[\kappa^{\rm abs}_\lambda + \kappa^{\rm scat}_\lambda\big]\,d(\rho x').
$$

We build it by cumulative integration over depth, for the full opacity (continuum + lines) and for the continuum alone, so we can form both the line spectrum and the continuum it sits on.

One asymmetry is worth flagging now: we *do* include scattering in this attenuation optical depth (it really does attenuate the beam), yet for this lecture we still approximate the *emissive* source function as $S_\lambda = B_\lambda$. That is an excellent approximation almost everywhere; the final section explains precisely where — the deepest line cores — it breaks down.""")

code(r'''def optical_depth(kappa):
    """Cumulative optical depth over column mass: tau[depth, wl]."""
    dtau = 0.5 * (kappa[1:] + kappa[:-1]) * np.diff(rhox)[:, None]     # trapezoid: mean kappa x column step between layers
    tau = np.empty_like(kappa)
    tau[0] = kappa[0] * rhox[0]                                        # seed the top layer with kappa * (column above it)
    tau[1:] = tau[0] + np.cumsum(dtau, axis=0)                         # accumulate inward, depth by depth
    return tau

tau_line = optical_depth(total_abs + total_scat)     # full optical depth: total extinction = absorption + scattering
tau_cont = optical_depth(cont_abs + cont_scat)        # continuum-only optical depth (no lines)
# sanity check: at 505 nm the scale should run from optically thin at the top to deep below the photosphere
print(f"at 505 nm, full optical depth spans {tau_line[:,2970].min():.1e} .. {tau_line[:,2970].max():.1e}")''')

md(r"""## The emergent flux

Now the formal solution, built in three short steps: the source function, the $E_2$ kernel, then the depth integral that combines them.

**A note on units.** We wrote the transfer equation with $\lambda$ subscripts ($F_\lambda$, $S_\lambda$, $B_\lambda$), but the code carries intensities per unit *frequency*, $B_\nu$, while still using wavelength as the grid coordinate. The same formal-solution equations hold with $\lambda$ replaced by $\nu$, and because the spectrum we report is a dimensionless *ratio* of two fluxes at the same wavelength, $F_\lambda/F_\lambda^{\rm cont} \equiv F_\nu/F_\nu^{\rm cont}$ — the per-Hz vs per-nm Jacobian cancels — the normalised spectrum is identical either way.

First the source function. In LTE it is the Planck function evaluated at each depth's temperature and each wavelength's frequency, $S_\lambda = B_\lambda(T)$.""")

code(r'''def planck_nu(nu, T):
    """Planck function per unit frequency, B_nu(T) [CGS]; pre-folded constants for speed."""
    x = H_C * nu / (K * T)                                          # dimensionless h*nu / kT
    return 1.47439e-2 * (nu/1e15)**3 * np.exp(-x) / (1.0 - np.exp(-x))

nu = C / (wl * 1e-7)                                   # frequency at each wavelength [Hz] (wl is in nm -> cm)
S = planck_nu(nu[None, :], T[:, None])                 # source function S = B(T), broadcast to (depth, wl)''')

md(r"""Next the kernel. The angle integral left us with the second exponential integral $E_2$; we supply it from scratch with the standard Abramowitz & Stegun rational approximations — first for $E_1$ (one polynomial for $x\le1$, a rational form for $x>1$), then $E_2(x) = e^{-x} - x\,E_1(x)$, with the limit $E_2(0)=1$ handled explicitly. No SciPy needed.""")

code(r'''def expint2(x):
    """Second exponential integral E2(x) = exp(-x) - x*E1(x), from scratch (no SciPy).
    E1 via Abramowitz & Stegun 5.1.53 (x<=1) and 5.1.56 (x>1); E2(0)=1."""
    x = np.asarray(x, float)
    xp = np.where(x > 0.0, x, 1.0)                      # placeholder at x=0 so log/division stay finite
    # E1 for small argument: polynomial fit minus the log singularity (A&S 5.1.53)
    a = (-0.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857)
    e1_small = (a[0] + xp*(a[1] + xp*(a[2] + xp*(a[3] + xp*(a[4] + xp*a[5]))))) - np.log(xp)
    # E1 for large argument: rational approximation times exp(-x)/x (A&S 5.1.56)
    a1, a2, b1, b2 = 2.334733, 0.250621, 3.330657, 1.681534
    e1_large = np.exp(-xp)/xp * (xp*xp + a1*xp + a2) / (xp*xp + b1*xp + b2)
    E1 = np.where(x <= 1.0, e1_small, e1_large)         # pick the branch element-wise
    return np.where(x > 0.0, np.exp(-x) - x*E1, 1.0)    # E2 from E1; force E2(0)=1''')

md(r"""Finally the depth integral. With $S$ and $E_2$ in hand, the emergent flux $F_\lambda = 2\pi\int S_\lambda E_2(\tau_\lambda)\,d\tau_\lambda$ is just a trapezoidal sum of $S\,E_2$ over the optical-depth grid, evaluated independently at every wavelength. Running it on the full opacity gives the line spectrum; on the continuum alone, the continuum flux. Their ratio is the **normalised spectrum** — the rectified line depths a spectroscopist actually measures, independent of any absolute flux calibration.""")

code(r'''def emergent_flux(tau):
    """F_lambda = 2*pi * integral S * E2(tau) over the optical-depth scale."""
    E2 = expint2(tau)                                  # evaluate the kernel on the tau grid (depth, wl)
    integrand = S * E2                                 # source function weighted by the kernel
    return 2*np.pi * np.trapezoid(integrand, tau, axis=0)   # integrate over depth at each wavelength

flux_line = emergent_flux(tau_line)                    # flux through the full (continuum + line) opacity
flux_cont = emergent_flux(tau_cont)                    # flux through the continuum alone
spectrum = flux_line / flux_cont                       # normalised (rectified) spectrum
reference = REF["flux_total"] / REF["flux_continuum"]  # the pykurucz reference, normalised the same way
rel = np.abs(spectrum - reference) / reference         # pixel-by-pixel relative difference
print(f"normalised spectrum vs reference:  median |rel diff| = {np.median(rel):.2e}   "
      f"max = {rel.max():.2e}  (the deepest line core)")''')

md(r"""**A part in a thousand, in the median.** Our from-scratch radiative transfer reproduces the reference solar spectrum to about a part in a thousand over most pixels across the whole window. The agreement loosens only in the deepest, scattering-dominated line cores; the single worst point reaches about ten percent — we will see why in a moment. First, the payoff: the spectrum itself.""")

code(r'''fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(wl, reference, color="0.6", lw=1.4, label="reference")
ax.plot(wl, spectrum, color="C3", lw=0.6, label="from scratch")
ax.set_xlabel("wavelength  [nm]"); ax.set_ylabel("normalised flux")
ax.set_title("The solar spectrum, 500–510 nm — built from scratch, matched to the reference")
ax.set_ylim(0, 1.05); ax.legend(loc="lower right"); fig.tight_layout(); plt.show()''')

md(r"""Every absorption line — a forest of neutral iron, vanadium, chromium, cobalt, and nickel — lands at the right wavelength with the right depth. The two curves are indistinguishable at this scale. We have computed a real stellar spectrum, from $T_{\rm eff}$ and $\log g$ and the laws of atomic physics, and matched a production code line for line.""")

md(r"""## The Eddington–Barbier relation

The formal solution has a famous approximation. If the source function is linear in optical depth, $S_\lambda \approx a + b\,\tau_\lambda$, the flux integral against the $E_2$ kernel evaluates *exactly* to $\pi(a + \tfrac23 b) = \pi\,S_\lambda(\tau_\lambda = 2/3)$. So whenever $S_\lambda$ varies slowly and roughly linearly, the emergent flux is close to $\pi$ times the source function at $\tau_\lambda = 2/3$:

$$
F_\lambda \approx \pi\,S_\lambda(\tau_\lambda = 2/3) = \pi\,B_\lambda\big(T(\tau_\lambda = 2/3)\big).
$$

This is the **Eddington–Barbier** relation, and it is why a spectrum is a thermometer of depth: at each wavelength we read the temperature of the layer where that wavelength's optical depth reaches $2/3$. In a line, the opacity is large, so $\tau_\lambda = 2/3$ is reached higher up in cooler gas, and the flux is lower. Let us check the approximation against the full integral.

![Line formation: in a line the extra opacity pushes $\tau=2/3$ up into a higher, cooler layer, so the flux drops — the Eddington–Barbier picture.](resources/figures/s6_rt.png)""")

code(r'''# Eddington-Barbier: read T where tau_lambda = 2/3 (interpolate the line tau scale onto 2/3 at each wl)
T_at_23 = np.array([np.interp(2/3, tau_line[:, k], T) for k in range(wl.size)])
flux_EB = np.pi * planck_nu(nu, T_at_23)               # approximate flux: pi * B at the tau=2/3 layer
# same shortcut for the continuum, so the EB spectrum is normalised exactly like the full one
spectrum_EB = flux_EB / (np.pi * planck_nu(nu, np.array([np.interp(2/3, tau_cont[:, k], T)
                                                         for k in range(wl.size)])))
print(f"Eddington-Barbier vs full formal solution: median |rel diff| = "
      f"{np.median(np.abs(spectrum_EB-spectrum)/spectrum):.2e}")
print(f"in a deep line core the tau=2/3 surface sits at T = {T_at_23.min():.0f} K "
      f"(vs {T_at_23.max():.0f} K in the continuum)")''')

md(r"""The Eddington–Barbier flux tracks the full solution to a few percent — good enough to read the physics off by eye, though we use the exact integral for the spectrum. It makes the line-formation picture concrete: in a deep core the $\tau=2/3$ surface climbs to a layer hundreds — in the deepest cores, more than a thousand — kelvin cooler than the continuum-forming layer, and the Planck function, steeply temperature-sensitive in the blue, drops accordingly.""")

# ── the one approximation, removed exactly next ─────────────────────────
md(r"""## The one approximation left: scattering

The formal solution itself is *exact* for any specified source function — the integral $F_\lambda = 2\pi\int S_\lambda E_2\,d\tau_\lambda$ is no approximation at all. Our *implementation* of it has made one remaining physical approximation: we set the source function equal to the Planck function, $S_\lambda = B_\lambda$. That LTE / pure-absorption choice is exact wherever true absorption dominates — almost everywhere — which is why the spectrum already matches to a part in a thousand. But in the very bottom of the deepest line cores, where the line opacity pushes $\tau=2/3$ up into the thin atmosphere, **scattering** (the Rayleigh and electron scattering of Lecture 3) becomes comparable to absorption, and the source function is pulled below the Planck function by photons that scatter and escape:

$$
S_\lambda = \frac{\kappa^{\rm abs}_\lambda\,B_\lambda + \kappa^{\rm scat}_\lambda\,J_\lambda}{\kappa^{\rm abs}_\lambda+\kappa^{\rm scat}_\lambda}.
$$

This depends on the mean intensity $J_\lambda$ (the angle-averaged radiation field), so $S_\lambda$ and $J_\lambda$ must be solved self-consistently with a $\Lambda$-iteration — repeatedly applying the operator $\Lambda$ that turns a source function into the mean intensity it produces — and it deepens the cores by up to about ten percent. The production code does exactly this with its **JOSH moment solver**: a method that solves for the angular *moments* of the field, such as $J_\lambda$, rather than only the outgoing intensity. That solver is the subject of the **next lecture**, where we rebuild it step by step and recover the spectrum to machine precision. The formal solution here gives the physics and the spectrum to a part in a thousand, and names precisely the one piece — scattering — that the exact engine adds.""")

md(r"""## Synthesis: the pipeline, complete

This is the summit of the course. Starting from two numbers — $T_{\rm eff}=5770\ \mathrm{K}$ and $\log g = 4.44$ — you built a grey **model atmosphere** (Lecture 1), solved its **equation of state** for the ionization and electron density (Lecture 2), computed the **continuous opacity** of H$^-$ (Lecture 3), the profile of a **single line** (Lecture 4) and the **forest of a million** (Lecture 5), and just now solved the **radiative transfer** that lets the light out. The result is a synthetic solar spectrum that reproduces a production code to a part in a thousand — every line in its place, carved by the atomic physics you implemented by hand.

Two things remain. First, the **next lecture** rebuilds the production radiative-transfer engine — the **JOSH moment solver** — which adds the scattering we just named and reproduces the spectrum to machine precision; this lecture gave the physics, the next gives the exact engine. Then we make the model itself honest: we took the grey temperature structure as given, and in Part IV we replace it, building the atmosphere self-consistently from **hydrostatic** and **radiative equilibrium** — and watch the spectrum settle onto the real Sun. Finally we add the **molecules** that bury the spectra of cool stars, and close the course.""")

md(r"""## Summary

- The plane-parallel transfer equation $\mu\,dI_\lambda/d\tau_\lambda = I_\lambda - S_\lambda$ has the formal solution $F_\lambda = 2\pi\int S_\lambda E_2(\tau_\lambda)\,d\tau_\lambda$; in LTE $S_\lambda = B_\lambda(T)$.
- Optical depth is the column-mass integral of the **total extinction** (absorption + scattering), $\tau_\lambda = \int(\kappa^{\rm abs}_\lambda+\kappa^{\rm scat}_\lambda)\,d(\rho x)$.
- The from-scratch spectrum reproduces the reference to a **median of $\sim5\times10^{-4}$** across $500$–$510\ \mathrm{nm}$.
- The **Eddington–Barbier** relation $F_\lambda\approx\pi B_\lambda(T(\tau_\lambda{=}2/3))$ makes a spectrum a depth thermometer.
- The formal solution's one approximation, $S_\lambda=B_\lambda$, is exact except in the deepest cores, where **scattering** deepens them by $\sim10\%$; the production **JOSH** solver (next lecture) treats it exactly and recovers the spectrum to machine precision.""")

md(r"""## Practice exercises

**1. The $E_2$ weighting.** Plot $E_2(\tau)$ from $\tau=0$ to $5$ and confirm it is largest at $\tau=0$ and decreases outward — it does *not* peak at $2/3$. Find the median depth of the kernel (where $\int_0^\tau E_2\,d\tau'$ reaches half its total) and note it is *not* $2/3$: the kernel's median and the Eddington–Barbier depth are different concepts. Then plug a linear source function $S = a + b\,\tau$ into $F = 2\pi\int S E_2\,d\tau$ and show analytically that the result equals $\pi\,S(\tau=2/3)$ — that is where $2/3$ actually comes from.

**2. A line-depth thermometer.** For the deepest line in the window, read $T(\tau_\lambda=2/3)$ at line centre and in the nearby continuum. Using the Planck function, predict the line's central depth and compare with the computed spectrum. Why are blue lines deeper than red ones at the same opacity contrast?

**3. Toward scattering.** Estimate the source function correction in a deep core by assuming $J_\lambda \approx \tfrac12 B_\lambda$ at the surface and forming $S_\lambda = (\kappa^{\rm abs}B + \kappa^{\rm scat}J)/(\kappa^{\rm abs}+\kappa^{\rm scat})$ with the scattering fraction from Lecture 3. By how much does the core darken, and does it move our spectrum toward the reference?""")

md(r"""## Further reading

- **Mihalas, D. (1978). *Stellar Atmospheres*, 2nd ed., Freeman.** Chapters 2–3 derive the transfer equation, the formal solution, and the exponential-integral kernels.
- **Rutten, R. J. (2003). *Radiative Transfer in Stellar Atmospheres* (Utrecht lecture notes).** A clear path from the transfer equation to the emergent spectrum, with the Eddington–Barbier relation.
- **Gray, D. F. (2005). *The Observation and Analysis of Stellar Photospheres*, 3rd ed.** Chapter 7 on radiative transfer and flux integrals.
- **Hubeny, I. & Mihalas, D. (2014). *Theory of Stellar Atmospheres*, Princeton.** The modern reference for scattering, the $\Lambda$-operator, and accelerated iteration.
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz*](https://arxiv.org/abs/2603.11693).** The implementation our reference spectrum is computed with.""")

nb = new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
