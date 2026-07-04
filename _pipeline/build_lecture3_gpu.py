#!/usr/bin/env python
"""Assemble content/Lecture3.ipynb (unexecuted). Execute + render via build.py.

Lecture 3 -- Continuous Opacity. These are self-contained graduate notes on
true absorption, scattering, H-minus continuum opacity, Rayleigh/Thomson scattering,
and the tabulated continuum construction.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture3.ipynb"

cells = []
def md(src): cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

md(r"""# Lecture 3 — Continuous Opacity

*Self-contained notes on H-minus absorption, scattering, and tabulated continuum opacity*

*Yuan-Sen Ting*

The continuum is the smooth opacity background on which spectral lines sit. In a solar-type photosphere it is set mainly by H-minus absorption, with a smaller scattering contribution from neutral hydrogen and free electrons. This lecture builds that continuum in two steps. First we write the analytic H-minus, Rayleigh, and Thomson terms so the physical scaling is clear. Then we replace the analytic fits with a tabulated continuum calculation used for detailed synthesis.

The calculations use PyTorch because every source naturally lives on a depth-by-wavelength grid: depth-dependent populations multiply wavelength-dependent cross-sections. A single expression therefore evaluates the whole grid at once and can run on a GPU when one is available.""")

md(r"""## Introduction

An opacity tells us how effectively matter removes photons from a beam. In stellar atmospheres we usually quote a **mass extinction coefficient**, $\kappa_\lambda$ in $\mathrm{cm^2\,g^{-1}}$, so the monochromatic optical-depth increment is

$$
d\tau_\lambda = \kappa_\lambda\,dm,
$$

where $dm=\rho\,dz$ is a column-mass increment. The continuum opacity is the slowly varying part of $\kappa_\lambda$: it changes with wavelength, but without the sharp resonance structure of spectral lines.

There are two ways photons are removed from a beam. **True absorption** converts photon energy into internal energy of the gas; in LTE its emission partner is tied to the Planck function. **Scattering** redirects photons without thermalizing them. The total continuous extinction is their sum,

$$
\kappa_{\lambda,\rm cont} = \kappa_{\lambda,\rm abs} + \kappa_{\lambda,\rm scat}.
$$

For a cool photosphere the dominant visible absorber is **H-minus**, the negative hydrogen ion: a neutral hydrogen atom with one extra, weakly bound electron. It is rare, but it has a large photodetachment cross-section, and its abundance scales with the free-electron density. Neutral-hydrogen Rayleigh scattering and electron Thomson scattering add a smaller scattering floor.""")

md(r"""## Atmosphere and wavelength grid

Load a solar atmosphere with temperature, density, electron density, neutral-hydrogen density, column mass, and a 500--510 nm wavelength grid. Depth-dependent quantities are stored as columns and wavelength-dependent quantities as rows, so multiplication broadcasts into a full $(\text{depth},\text{wavelength})$ opacity grid.""")

code(r'''import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt

# Execution machinery only: the opacity equations below are written as tensor operations.
if torch.backends.mps.is_available():
    DEVICE, DTYPE = torch.device("mps"), torch.float32
elif torch.cuda.is_available():
    DEVICE, DTYPE = torch.device("cuda"), torch.float32
else:
    DEVICE, DTYPE = torch.device("cpu"), torch.float64

def dev(x):
    """Convert an input array to the working array type."""
    return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=DEVICE)

plt.rcParams.update({
    "figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
})

REF = np.load(pathlib.Path("..") / "reference" / "L3.npz")''')

md(r"""Unpack the grids and populations, reshaping the depth-dependent quantities to a column and the wavelength grid to a row. Every opacity below is then a $(\text{depth}, \text{wavelength})$ array: a population column times a cross-section row, divided by the mass density.""")

code(r'''# The data bundle keeps compact historical keys; translate once into taught names.
wavelength_nm = dev(REF["wl"])[:, None].T            # (1,nw) [nm]
temperature = dev(REF["T"])[:, None]                 # (nd,1) [K]
electron_density = dev(REF["n_e"])[:, None]          # (nd,1) [cm^-3]
mass_density = dev(REF["rho"])[:, None]              # (nd,1) [g cm^-3]
neutral_hydrogen_density = dev(REF["nHI"])[:, None]  # (nd,1) neutral H number density [cm^-3]
n_wavelength = wavelength_nm.shape[1]

# Rosseland optical depth and column mass, used for plotting / formation depth
rosseland_optical_depth, column_mass_host = REF["tau"], REF["rhox"]

# physical constants, CGS: Planck, c, Boltzmann
H, C, K = 6.62607015e-27, 2.99792458e10, 1.380649e-16
# eV-per-K conversion; Saha (2pi m_e k/h^2)^3/2 prefactor
KEV, SAHA = 1.0/11604.5, 2.4148e15

# frequency [Hz] at each wavelength (a row, broadcasts over depth)
frequency_hz = C / (wavelength_nm * 1e-7)

print(f"continuum grid: {REF['absorption'].shape[0]} layers x {n_wavelength} wavelengths, "
      f"{float(wavelength_nm[0, 0]):.0f}-{float(wavelength_nm[0, -1]):.0f} nm")''')

md(r"""## The stimulated-emission factor

Every true-absorption coefficient carries a stimulated-emission correction. A photon can induce the inverse transition as well as be absorbed, so the net absorption is multiplied by

$$
\big(1 - e^{-h\nu/kT}\big),
$$

near $1$ in the blue, dropping toward the infrared where $h\nu \lesssim kT$. Scattering, which redirects photons without converting their energy into heat, carries no such factor.""")

code(r'''# stimulated-emission correction, (nd, nw): depth column temperature x wavelength row frequency.
stimulated_emission_factor = 1.0 - torch.exp(-H * frequency_hz / (K * temperature))
print(f"stimulated-emission factor at 505 nm: "
      f"{float(stimulated_emission_factor[:,100].min()):.3f} (hot, deep) .. "
      f"{float(stimulated_emission_factor[:,100].max()):.3f} (cool)")''')

md(r"""## How much H-minus is there? The Saha balance

In a cool star the dominant continuous absorber is **H-minus**, the negative hydrogen ion: a neutral hydrogen atom holding a second, weakly bound electron with binding energy $\chi = 0.754\ \mathrm{eV}$. It is fragile and rare, but neutral hydrogen is abundant and metal ionization supplies enough free electrons that H-minus dominates the visible continuous opacity of solar-type photospheres. Its abundance follows a Saha balance, with the small detachment energy in place of a usual ionization potential:

$$
n(\mathrm{H}^-) = \frac{n(\mathrm{H\,I})\,n_e}{4\,(2.4148\times10^{15})\,T^{3/2}}\;e^{+\chi/kT}.
$$

The factor of four is statistical-weight accounting: two spin states for the free electron times the neutral-hydrogen ground-state weight, divided by the closed-shell H-minus weight. The positive exponent says H-minus is favoured at low temperature, and the explicit $n_e$ is why H-minus opacity tracks electron density and therefore metallicity. We compute the H-minus density once; both opacity channels reuse it.

![The H-minus ion, a hydrogen atom holding a second weakly bound electron, is the dominant continuous absorber in cool stars; a photon detaches the electron, and its abundance tracks the electron density.](resources/figures/s3_hminus.png)""")

code(r'''chi_Hminus = 0.754   # H-minus electron binding energy [eV]

# the Saha balance, solved for the H-minus density [cm^-3] — a (nd,1) depth column
n_Hminus = (
    neutral_hydrogen_density * electron_density
    * torch.exp(chi_Hminus / (KEV * temperature))
    / (4.0 * SAHA * temperature**1.5)
)

jp = int(torch.argmin(torch.abs(torch.as_tensor(rosseland_optical_depth, dtype=DTYPE, device=DEVICE) - 2/3)))
print(f"n(H-minus)/n(H I) at the photosphere: {float((n_Hminus/neutral_hydrogen_density)[jp,0]):.2e}  "
      f"(about two H-minus ions per billion H atoms)")''')

md(r"""## H-minus bound-free and free-free opacity

H-minus absorbs by two channels. **Bound-free** absorption, or photodetachment, ejects the weakly bound electron; it has a threshold at $\lambda_0 = 1.6419\ \mathrm{\mu m}$ and peaks in the red. **Free-free** absorption is absorption by a passing electron in the field of a neutral H atom; it has no threshold and rises into the infrared. We use the analytic fits of **John (1988)**.

The bound-free cross-section, with $\lambda$ in microns and $f \equiv 1/\lambda - 1/\lambda_0$, is $\sigma_{\rm bf} = 10^{-18}\,\lambda^3\,f^{3/2}\sum_{n=0}^{5} C_n\, f^{n/2}$, *zero past the threshold* ($\lambda \geq \lambda_0$, where $f \leq 0$). The threshold needs one careful numerical step: fractional powers require a nonnegative base. With $x = \sqrt{\max(0,\,f)}$, the cross-section becomes

$$
\sigma_{\rm bf} = 10^{-18}\,\lambda^3\,x^3\sum_{n=0}^{5} C_n x^n,
$$

a Horner polynomial that is exactly zero past threshold because $x=0$ there. We also use the algebraically stable form $f = (\lambda_0-\lambda)/(\lambda_0\lambda)$ near the edge.""")

code(r'''# H-minus bound-free (John 1988) — clamp-then-Horner evaluation (branchless, NaN-free).
lam_um = wavelength_nm * 1e-3        # (1, nw) [um]
lam0 = 1.6419                        # threshold [um] (0.754 eV)

# stable form of f = 1/lam - 1/lam0 (avoids fp32 cancellation near the edge), then clamp >= 0
f = torch.clamp((lam0 - lam_um) / (lam0 * lam_um), min=0.0)
x = torch.sqrt(f)                    # x = sqrt(max(0,f)); EXACTLY 0 where lam >= lam0 -> no NaN

# sum_n C_n f^(n/2) = sum_n C_n x^n  (Horner); sigma = 1e-18 lam^3 x^3 * poly  (x^3 = f^1.5)
C_bf = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982]
poly = C_bf[5]
for c in reversed(C_bf[:5]):         # JUSTIFIED-LOOP: six fixed polynomial coefficients, no data-axis loop.
    poly = poly * x + c
sigma_bf = 1e-18 * lam_um**3 * x**3 * poly       # (1, nw) [cm^2]; zero past threshold by construction

# opacity per gram, with the stimulated-emission factor [cm^2/g] — broadcasts to (nd, nw)
kappa_bf = n_Hminus * sigma_bf * stimulated_emission_factor / mass_density''')

md(r"""The free-free coefficient is John's second polynomial: a double power series in $\theta = 5040/T$ and in inverse wavelength, returning absorption per neutral H atom per unit electron pressure $P_e = n_e kT$ (the stimulated-emission factor is already folded into the fit). We evaluate the inverse-wavelength terms with explicit reciprocal powers and sum the five $\theta$-orders — the whole thing a $(\text{nd}, \text{nw})$ tensor: $\theta$ is a depth column, the wavelength terms a row.""")

code(r'''# H-minus free-free (John 1988, lambda > 0.3645 um branch), per H I atom per unit P_e.
A=[0,2483.346,-3449.889,2200.040,-696.271,88.283]; B=[0,285.827,-1158.382,2427.719,-1841.400,444.517]
Cc=[0,-2054.291,8746.523,-13651.105,8624.970,-1863.864]; D=[0,2827.776,-11485.632,16755.524,-10051.530,2095.288]
E=[0,-1341.537,5303.609,-7510.494,4400.067,-901.788]; F=[0,208.952,-812.939,1132.738,-655.020,132.985]

theta = 5040.0 / temperature         # (nd,1) John temperature variable (depth column)
inv = 1.0 / lam_um                   # (1,nw) inverse wavelength (row)
kff = torch.zeros_like(stimulated_emission_factor)  # (nd, nw)
for n in range(1, 6):                # JUSTIFIED-LOOP: five fixed polynomial orders, not depth or wavelength.
    term = A[n]*lam_um**2 + B[n] + Cc[n]*inv + D[n]*inv**2 + E[n]*inv**3 + F[n]*inv**4
    kff = kff + theta**((n+1)/2.0) * term

P_e = electron_density * K * temperature          # electron pressure [dyn cm^-2] (depth column)
kappa_ff = 1e-29 * kff * P_e * neutral_hydrogen_density / mass_density  # [cm^2/g]
kappa_Hminus = kappa_bf + kappa_ff               # total analytic H-minus absorption (nd, nw)''')

md(r"""**Read the H-minus opacity.** The analytic model is deliberately simple: it keeps H-minus bound-free and free-free absorption, the two terms that explain the visible continuum of a cool star. The code below records the spectrum-forming layers and prints the relative size of the two H-minus channels near the photosphere.""")

code(r'''def to_np(t):
    """Tensor -> NumPy for plotting and scalar summaries."""
    return t.detach().cpu().to(torch.float64).numpy() if torch.is_tensor(t) else np.asarray(t, float)

# H-minus dominates where the optical spectrum forms; keep that layer mask for summaries.
form = (rosseland_optical_depth > 1e-3) & (rosseland_optical_depth < 3.0)
absn = to_np(kappa_Hminus)
bf_share = float(kappa_bf[jp,100] / kappa_Hminus[jp,100])
print(f"H-minus absorption at 505 nm, tau=2/3: {float(kappa_Hminus[jp,100]):.3e} cm^2/g")
print(f"bound-free supplies {100*bf_share:.1f}% of that H-minus opacity; free-free supplies {100*(1-bf_share):.1f}%")''')

md(r"""At 500--510 nm the bound-free channel carries most of the H-minus absorption, because these photons have enough energy to detach the weakly bound electron. The free-free channel grows toward the infrared, where a passing electron can absorb photon energy while remaining free.""")

md(r"""## Scattering: Rayleigh beats Thomson

The textbook reflex is **Thomson** scattering off free electrons ($\sigma_T = 0.6653\times10^{-24}\ \mathrm{cm^2}$). But in a cool photosphere the free electrons are scarce while neutral hydrogen is everywhere, so **Rayleigh** scattering off the bound electrons of neutral H — the same $\lambda^{-4}$ scattering that makes the sky blue — dominates. We use the Dalgarno polarizability fit, with $\lambda$ in ångström, and add Thomson. Both are pure wavelength rows scaled by depth populations; we evaluate the inverse powers with explicit reciprocals.""")

code(r'''lamA = wavelength_nm * 10.0          # (1, nw) [angstrom]
invA = 1.0 / lamA
inv2 = invA*invA; inv4 = inv2*inv2; inv6 = inv4*inv2; inv8 = inv4*inv4

# Rayleigh off neutral H (Dalgarno) + grey Thomson off electrons — broadcast to (nd, nw)
sigma_Ray = 5.799e-13*inv4 + 1.422e-6*inv6 + 2.784*inv8         # [cm^2] per H I atom
kappa_Ray = sigma_Ray * neutral_hydrogen_density / mass_density
kappa_Thomson = 0.6653e-24 * electron_density / mass_density * torch.ones_like(frequency_hz)
kappa_scat = kappa_Ray + kappa_Thomson
kappa_total = kappa_Hminus + kappa_scat

scatn = to_np(kappa_scat)
print(f"at the photosphere, Rayleigh is {float(kappa_Ray[jp,100]/kappa_Thomson[jp,100]):.0f}x Thomson; "
      f"scattering is {float(100*kappa_scat[jp,100]/kappa_Hminus[jp,100]):.1f}% of the H-minus absorption")''')

md(r"""Rayleigh outweighs Thomson several-fold at the $\tau=2/3$ layer (the ratio climbs higher in the cooler layers above, where free electrons grow even scarcer), and scattering as a whole is only about a percent of the H-minus absorption — the solar optical continuum is an *absorption* continuum, which is why its source function stays close to the Planck function. In the ultraviolet, where $\lambda^{-4}$ Rayleigh climbs and metal photoionization switches on, the balance shifts; in hot stars, where hydrogen is ionized, Thomson takes over.""")

md(r"""## The total continuum and where it forms

Add absorption and scattering for the total continuous extinction, and convert it to optical depth. Here `column_mass_host` is the inward column mass $m = \int\rho\,dz$, so $d\tau_\lambda = \kappa_\lambda\,dm$; integrating the continuum opacity over the column mass tells us the depth from which the continuum escapes.""")

code(r'''totn = to_np(kappa_total)
# continuum optical depth at 505 nm, integrated over column mass (trapezoid)
k505 = totn[:, 100]
tau_cont = np.zeros_like(column_mass_host)
tau_cont[1:] = np.cumsum(0.5*(k505[1:]+k505[:-1]) * np.diff(column_mass_host))
j23 = int(np.argmin(np.abs(tau_cont - 2/3)))
print(f"the 505 nm continuum reaches tau=2/3 at T = {REF['T'][j23]:.0f} K  "
      f"(log tau_Ross = {np.log10(rosseland_optical_depth[j23]):.2f})")''')

md(r"""Plot the opacity against depth (absorption versus scattering) and the integrated continuum optical depth that locates the $\tau = 2/3$ surface.""")

code(r'''logtau = np.log10(rosseland_optical_depth)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))
ax[0].plot(logtau, np.log10(totn[:,100]), color="C0", label="total")
ax[0].plot(logtau, np.log10(to_np(kappa_Hminus)[:,100]), "--", color="C3", label="H-minus absorption")
ax[0].plot(logtau, np.log10(scatn[:,100]), ":", color="C2", label="scattering")
ax[0].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[0].set_ylabel(r"$\log_{10}\kappa_{505}$  [cm$^2$/g]")
ax[0].set_title("Continuum opacity vs depth"); ax[0].legend()

ax[1].plot(logtau, tau_cont, color="C0")
ax[1].axhline(2/3, ls="--", color="0.5", lw=1, label=r"$\tau_{505}=2/3$")
ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$")
ax[1].set_ylabel(r"continuum optical depth $\tau_{505}$"); ax[1].set_title("Where the continuum forms")
ax[1].legend()
fig.tight_layout(); plt.show()''')

md(r"""The analytic continuum gives the physical picture: H-minus sets the absorption scale, Rayleigh and Thomson add a small scattering contribution, and the depth of formation follows from integrating $\kappa_\lambda$ over column mass. A detailed synthesis still needs more: tabulated H-minus cross-sections, hydrogenic bound-free and free-free terms, helium continua, molecular and metal edges, and the wavelength reconstruction around continuum edges. That is the purpose of the tabulated calculation below.""")


md(r"""## From analytic fits to tabulated opacity

The analytic H-minus model explains the continuum, but detailed synthesis uses measured or carefully computed tables wherever the cross-sections have structure. The tabulated calculation keeps the same physical sources and adds three pieces of realism.

First, H-minus bound-free absorption is read from a table rather than from the John polynomial. Second, the budget includes the smaller sources that matter at the percent level: H I bound-free and free-free opacity, molecular hydrogen ion opacity, helium-minus opacity, carbon/magnesium/aluminium/silicon edges, helium continua, hot-star free-free terms, Rayleigh scattering, and Thomson scattering. Third, the opacity is sampled on a small **edge-triplet frequency grid** and reconstructed at the desired wavelengths with a 3-point Lagrange interpolation in $\log_{10}\kappa$.

The code below builds that tabulated path from named source terms. Short loops remain only over small fixed physics tables, such as a handful of metal edges or hydrogenic levels.""")

md(r"""## Constants and cross-section tables

Load the atmosphere/equation-of-state arrays and the cross-section tables. The tables include discrete brackets in temperature, frequency, and edge interval. Those brackets are part of the data model: once the bracket is chosen, the opacity itself is still computed from physical source terms.""")

code(r'''# Atmosphere/EOS, wavelength grid, and continuum cross-section tables.
A_np = np.load(pathlib.Path("..") / "reference" / "atmosphere.npz", allow_pickle=True)
D_np = np.load(pathlib.Path("..") / "reference" / "diag.npz")
KT_np = np.load(pathlib.Path("..") / "reference" / "kapp_tables.npz")

# Tabulated opacity arrays. These tables are small and include discrete bracket choices.
EDEV, EDTYPE = torch.device("cpu"), torch.float64

def dev64(x):
    """Table/atmosphere array -> the tabulated-opacity dtype."""
    return torch.as_tensor(x, dtype=EDTYPE, device=EDEV)

wlk = dev64(D_np["wavelength"])                         # [nm], synthesis grid

Tk = dev64(A_np["temperature"])
rho_k = torch.clamp(dev64(A_np["mass_density"]), min=1e-30)
ne_k = dev64(A_np["electron_density"])
n_layers_k = Tk.shape[0]

print(f"tabulated continuum grid: {n_layers_k} layers x {wlk.shape[0]} wavelengths, "
      f"{float(wlk[0]):.1f}-{float(wlk[-1]):.1f} nm")''')

md(r"""The constants are repeated with the names used in the tabulated formulas so the correspondence to each opacity term is easy to audit. They are the same CGS values already used above.""")

code(r'''# CGS constants used by the tabulated opacity formulas.
C_LIGHT_CM_k = 2.99792458e10
C_LIGHT_NM_k = 2.99792458e17
H_PLANCK_k = 6.62607015e-27
K_BOLTZ_k = 1.380649e-16
KBOLTZ_EV_k = 8.6171e-5
RYDBERG_CM_k = 109677.576
LN10_k = torch.log(torch.tensor(10.0, dtype=EDTYPE, device=EDEV))
print("tabulated-opacity constants loaded")''')

md(r"""The hydrogen-group tables are the most important ones: H-minus bound-free, H-minus free-free, and Karzas--Latter H I bound-free.""")

code(r'''# Hydrogen / H-minus tables.
FREQ_LOG_t = dev64(KT_np["FREQ_LOG"])
XN_LOG_t = dev64(KT_np["XN_LOG"])
XL_LOG_ARRAY_t = dev64(KT_np["XL_LOG_ARRAY"])
EKARSAS_t = dev64(KT_np["EKARSAS"])
HMINOP_WBF_t = dev64(KT_np["HMINOP_WBF"])
HMINOP_BF_t = dev64(KT_np["HMINOP_BF"])
HMINOP_WAVEK_t = dev64(KT_np["HMINOP_WAVEK"])
HMINOP_THETAFF_t = dev64(KT_np["HMINOP_THETAFF"])
HMINOP_FFBEG_t = dev64(KT_np["HMINOP_FFBEG"])
HMINOP_FFEND_t = dev64(KT_np["HMINOP_FFEND"])
print("hydrogen and H-minus tables loaded")''')

md(r"""The second group supplies the scattering factors, Coulomb free-free Gaunt factors, metal and hot-star opacity tables, and the hydrogen partition-function inputs.""")

code(r'''# Scattering / free-free / minor tables.
HRAYOP_GAVRILAM_t = dev64(KT_np["HRAYOP_GAVRILAM"])
COULFF_Z4LOG_t = dev64(KT_np["COULFF_Z4LOG"])
COULFF_A_TABLE_t = dev64(KT_np["COULFF_A_TABLE"])
HOTOP_TRANSITIONS_t = dev64(KT_np["HOTOP_TRANSITIONS"])
SI2_PEACH_t = dev64(KT_np["_SI2OP_PEACH"])
SI2_FREQSI_t = dev64(KT_np["_SI2OP_FREQSI"])
SI2_FLOG_t = dev64(KT_np["_SI2OP_FLOG"])
SI2_TLG_t = dev64(KT_np["_SI2OP_TLG"])
H_ENERGY_EV_t = dev64(KT_np["H_ENERGY_CM"]) / 8065.479
H_STAT_WEIGHT_t = dev64(KT_np["H_STAT_WEIGHT"])
print("scattering, COULFF, HOTOP, Si II, and partition tables loaded")''')

md(r"""## The edge-triplet frequency grid and the 3-point interpolation

The continuum is smooth between photo-ionization edges, but it can change abruptly at an edge. For each edge interval the tabulated calculation samples the opacity at three frequencies: just inside the high-frequency edge, at the wavelength midpoint, and just inside the low-frequency edge. The factor `1.0000001` nudges the samples away from the discontinuity so the sampled point does not sit exactly on an edge.""")

code(r'''frqedg_t = dev64(A_np["frqedg"])
wledge_signed_t = dev64(A_np["wledge"])
wledge_t = torch.abs(wledge_signed_t)
n_edges_k = frqedg_t.shape[0]

freq_hi = torch.abs(frqedg_t[:-1]) / 1.0000001
freq_mid = C_LIGHT_NM_k / ((torch.abs(wledge_signed_t[:-1]) + torch.abs(wledge_signed_t[1:])) * 0.5)
freq_lo = torch.abs(frqedg_t[1:]) * 1.0000001
freqset_t = torch.stack((freq_hi, freq_mid, freq_lo), dim=1).reshape(-1)

edge_idx_t = torch.clamp(torch.searchsorted(wledge_t, torch.abs(wlk), right=True) - 1, 0, n_edges_k - 2)
used_edges_t = torch.unique(edge_idx_t)
sel_t = (3 * used_edges_t[:, None] + torch.arange(3, device=EDEV)[None, :]).reshape(-1)
freq_sel_t = freqset_t.index_select(0, sel_t.to(torch.int64))

print(f"{int(n_edges_k)} edges -> {int(freqset_t.shape[0])} sample frequencies in the full grid")
print(f"this window uses {int(used_edges_t.shape[0])} edge interval(s), {int(freq_sel_t.shape[0])} sampled frequencies")''')

md(r"""## The populations the opacity reads

Each continuum source is a cross-section times a population divided by the mass density. The atmosphere file stores the equation-of-state output in several population conventions: hydrogen ground-state populations in `xnfph`, stage totals for Rayleigh, helium stages, metal bound-free populations, and the charge-weighted sums needed by the hot-star free-free term.""")

code(r'''pop_t = dev64(A_np["population_per_ion"])

xnfph_t = dev64(A_np["xnfph"])
xnf_h_t = dev64(A_np["xnf_h"])
xnf_he1_t = dev64(A_np["xnf_he1"])
xnf_he2_t = dev64(A_np["xnf_he2"])
xnfpc_t = dev64(A_np["xnfpc"])
xnfpmg_t = dev64(A_np["xnfpmg"])
xnfpal_t = dev64(A_np["xnfpal"])
xnfpsi_t = dev64(A_np["xnfpsi"])
xnfpfe_t = dev64(A_np["xnfpfe"])

print(f"ground-state H I population: {float(xnfph_t[0,0]):.3e} .. {float(xnfph_t[-1,0]):.3e} cm^-3")''')

md(r"""The helium tables use two closely related population conventions: stage totals for the ionization balance, and lower-state populations for selected helium continuum branches. Helium-minus free-free, He Rayleigh scattering, and the He I/II bound-free routines need the latter, so both are named explicitly rather than treated as interchangeable.

The hot-star term is tiny in this solar optical window, but it belongs to the complete continuum budget. It uses per-transition populations and charge-weighted free-free population sums.""")

code(r'''# Helium lower-state populations used by He- free-free, He Rayleigh, and He bound-free terms.
xnf_he1_mode11_t = pop_t[:, 0, 1]
xnf_he2_mode11_t = pop_t[:, 1, 1]
xnf_he3_mode11_t = pop_t[:, 2, 1]

# HOTOP transition populations: vectorized gather of the fixed element/stage slices.
hotop_xnfp_t = torch.zeros((n_layers_k, 21), dtype=EDTYPE, device=EDEV)
hotop_xnfp_t[:, 0:4] = pop_t[:, 0:4, 5]
hotop_xnfp_t[:, 4:9] = pop_t[:, 0:5, 6]
hotop_xnfp_t[:, 9:15] = pop_t[:, 0:6, 7]
hotop_xnfp_t[:, 15:21] = pop_t[:, 0:6, 9]

elem_hot = torch.tensor([5, 6, 7, 9, 11, 13, 15, 25], dtype=torch.int64, device=EDEV)
stage_hot = torch.arange(1, 6, dtype=torch.int64, device=EDEV)
charge2 = (stage_hot.to(EDTYPE) ** 2)[None, :, None]
xnf_sumqq_t = torch.sum(pop_t[:, stage_hot[:, None], elem_hot[None, :]] * charge2, dim=2)
print("HOTOP population vectors built")''')

md(r"""## Table interpolation and the Karzas--Latter lookup

Table interpolation is part of the opacity model. A local three-point parabolic reconstruction is used for the H-minus bound-free table; linear interpolation is used where the source table is smooth enough that two neighboring entries define the local trend. The Karzas--Latter hydrogenic lookup is another table interpolation, but in threshold frequency rather than wavelength.""")

code(r'''def linter_torch(xold, yold, xnew):
    """Linear interpolation/extrapolation, vectorized over xnew; xold is increasing."""
    idx = torch.clamp(torch.searchsorted(xold, xnew, right=True), 1, xold.shape[0] - 1)
    x0 = xold.index_select(0, idx - 1); x1 = xold.index_select(0, idx)
    y0 = yold.index_select(0, idx - 1); y1 = yold.index_select(0, idx)
    w = (xnew - x0) / torch.where(torch.abs(x1 - x0) < 1e-40, torch.ones_like(x1), x1 - x0)
    return y0 + (y1 - y0) * w

def linter_matrix_torch(xold, yold, xnew):
    """Interpolate yold[x, column] at many xnew; returns (len(xnew), ncolumn)."""
    idx = torch.clamp(torch.searchsorted(xold, xnew, right=True), 1, xold.shape[0] - 1)
    x0 = xold.index_select(0, idx - 1); x1 = xold.index_select(0, idx)
    y0 = yold.index_select(0, idx - 1); y1 = yold.index_select(0, idx)
    w = ((xnew - x0) / torch.where(torch.abs(x1 - x0) < 1e-40, torch.ones_like(x1), x1 - x0))[:, None]
    return y0 + (y1 - y0) * w

def parabolic_table_torch(xold, yold, xnew):
    """Local 3-point parabolic table interpolation, vectorized over xnew."""
    n = xold.shape[0]
    j = torch.clamp(torch.searchsorted(xold, xnew, right=True), 1, n - 2)
    i0, i1, i2 = j - 1, j, j + 1
    x0 = xold.index_select(0, i0); x1 = xold.index_select(0, i1); x2 = xold.index_select(0, i2)
    y0 = yold.index_select(0, i0); y1 = yold.index_select(0, i1); y2 = yold.index_select(0, i2)
    l0 = (xnew - x1) * (xnew - x2) / ((x0 - x1) * (x0 - x2))
    l1 = (xnew - x0) * (xnew - x2) / ((x1 - x0) * (x1 - x2))
    l2 = (xnew - x0) * (xnew - x1) / ((x2 - x0) * (x2 - x1))
    return y0*l0 + y1*l1 + y2*l2''')

md(r"""The Karzas--Latter hydrogenic cross-section is a lookup in descending $\log_{10}\nu$ columns. The vectorized version below forms all comparisons at once and uses `argmax`/`gather` rather than a binary-search loop over frequencies.""")

code(r'''def xkarsas_vec(freq_vec, zeff_squared, n, ell):
    """Karzas-Latter hydrogenic bound-free cross-section for one level.

    Parameters are the sampled frequencies, effective charge squared, principal
    quantum number, and angular-momentum selector used by the table.
    The return value is a length-``nf`` cross-section row.  For tabulated
    ``n <= 15`` levels this gathers from the Karzas--Latter log tables; for
    higher levels it uses the asymptotic ``n=15`` table in the scaled energy
    variable.  Frequencies below threshold are explicitly zeroed.
    """
    freq_log = torch.log10(freq_vec / zeff_squared)
    if n <= 15:
        column = FREQ_LOG_t[:, n - 1]
        values = XN_LOG_t[:, n - 1] if (ell >= n or n > 6) else XL_LOG_ARRAY_t[ell, n - 1, :]
        active = freq_log >= column[-1]
        # The table is ordered in descending log-frequency.  Compare every
        # requested frequency against all interior breakpoints, then gather the
        # first bracket index without a Python loop over frequencies.
        cmp = freq_log[:, None] > column[None, 1:]
        any_cmp = torch.any(cmp, dim=1)
        first = torch.argmax(cmp.to(torch.int64), dim=1) + 1
        idx = torch.where(any_cmp, first, torch.full_like(first, column.shape[0]))
        last_val = torch.exp(values[-1] * LN10_k) / zeff_squared
        im = torch.clamp(idx, 1, column.shape[0] - 1)
        denom = column.index_select(0, im - 1) - column.index_select(0, im)
        w = (freq_log - column.index_select(0, im)) / torch.where(torch.abs(denom) < 1e-15, torch.ones_like(denom), denom)
        xval = (values.index_select(0, im - 1) - values.index_select(0, im)) * w + values.index_select(0, im)
        val = torch.where(idx >= column.shape[0], last_val, torch.exp(xval * LN10_k) / zeff_squared)
        return torch.where(active, val, torch.zeros_like(freq_vec))

    inv_n2 = 1.0 / (n * n)
    ryd_c = RYDBERG_CM_k * C_LIGHT_CM_k
    f0 = torch.log10(torch.tensor(ryd_c * inv_n2, dtype=EDTYPE, device=EDEV))
    active = freq_log >= f0
    egrid = EKARSAS_t[1:28]
    fcur = torch.log10((egrid + inv_n2) * ryd_c)
    # Same bracket search as above, but in the scaled high-n energy grid.
    cmp = freq_log[:, None] > fcur[None, :]
    any_cmp = torch.any(cmp, dim=1)
    first = torch.argmax(cmp.to(torch.int64), dim=1) + 1
    im = torch.clamp(first, 1, 27)
    fprev = torch.where(im == 1, f0, torch.log10((EKARSAS_t.index_select(0, im - 1) + inv_n2) * ryd_c))
    fc = torch.log10((EKARSAS_t.index_select(0, im) + inv_n2) * ryd_c)
    denom = fprev - fc
    w = (freq_log - fc) / torch.where(torch.abs(denom) == 0.0, torch.ones_like(denom), denom)
    xval = (XN_LOG_t.index_select(0, im - 1)[:, 14] - XN_LOG_t.index_select(0, im)[:, 14]) * w + XN_LOG_t.index_select(0, im)[:, 14]
    val = torch.where(any_cmp, torch.exp(xval * LN10_k) / zeff_squared,
                      torch.exp(XN_LOG_t[28, 14] * LN10_k) / zeff_squared)
    return torch.where(active, val, torch.zeros_like(freq_vec))''')

md(r"""The Coulomb free-free Gaunt factor is the other important interpolator. It maps charge, temperature, and frequency into the `COULFF_A_TABLE` grid and performs the bilinear blend as a single `(frequency, depth)` tensor expression.""")

code(r'''def coulff_table_torch(nz, freq_vec, temp, tlog, freqlg_override=None):
    """Interpolate the COULFF free-free Gaunt-factor table.

    ``nz`` selects the ionic charge row, ``freq_vec`` is the sampled frequency
    vector, and ``temp``/``tlog`` are depth vectors.  The returned tensor has
    shape ``(nfreq, nlayer)`` so callers can transpose it into the usual
    ``(depth, frequency)`` opacity shape.  ``freqlg_override`` handles the
    He II table convention where the logarithmic lookup frequency is not the
    same as the explicit frequency in the rest of the formula.
    """
    if nz < 1 or nz > 6:
        return torch.ones((freq_vec.shape[0], temp.shape[0]), dtype=EDTYPE, device=EDEV)
    A_tab = COULFF_A_TABLE_t
    z4log = COULFF_Z4LOG_t[nz - 1]
    freqlg = torch.log(freq_vec) if freqlg_override is None else torch.full_like(freq_vec, freqlg_override)
    gamlog = 10.39638 - tlog / 1.15129 + z4log
    hvktlg = (freqlg[:, None] - tlog[None, :]) / 1.15129 - 20.63764
    igam = torch.clamp((gamlog + 7.0).to(torch.int64), 1, 10)
    ihvkt = torch.clamp((hvktlg + 9.0).to(torch.int64), 1, 11)
    p = gamlog - (igam.to(EDTYPE) - 7.0)
    q = hvktlg - (ihvkt.to(EDTYPE) - 9.0)
    ig = igam - 1; ih = ihvkt - 1
    # Broadcast the depth-only gamma coordinate over the frequency axis, then
    # use flattened gathers because advanced two-axis indexing is backend-fragile.
    ig_b = ig[None, :].expand(freq_vec.shape[0], -1)
    p_b = p[None, :].expand(freq_vec.shape[0], -1)

    def gather(ri, ci):
        return A_tab.reshape(-1)[ri * A_tab.shape[1] + ci]

    a00 = gather(ig_b, ih)
    ihp = torch.minimum(ih + 1, torch.tensor(10, dtype=torch.int64, device=EDEV))
    igp = torch.minimum(ig_b + 1, torch.tensor(11, dtype=torch.int64, device=EDEV))
    a01 = torch.where(ihvkt < 11, gather(ig_b, ihp), a00)
    a10 = torch.where(igam[None, :] < 10, gather(igp, ih), a00)
    a11 = torch.where((igam[None, :] < 10) & (ihvkt < 11), gather(igp, ihp), a00)
    return (1.0 - p_b) * ((1.0 - q) * a00 + q * a01) + p_b * ((1.0 - q) * a10 + q * a11)''')

md(r"""The Planck function gives the LTE thermal radiation scale, and the hydrogen partition function converts total neutral hydrogen into the ground-state population used by Rayleigh scattering. Both are small helpers, but they keep the opacity formulas tied to their physical inputs.""")

code(r'''def hydrogen_partition_torch(temp):
    kt = KBOLTZ_EV_k * temp
    return torch.sum(H_STAT_WEIGHT_t[:6][None, :] * torch.exp(-H_ENERGY_EV_t[:6][None, :] / kt[:, None]), dim=1)

def planck_nu_torch(freq, temp):
    x = H_PLANCK_k * freq[None, :] / (K_BOLTZ_k * temp[:, None])
    const = 2.0 * H_PLANCK_k / C_LIGHT_CM_k**2
    safe = torch.where(x < 1e-6, torch.ones_like(x), x)
    bnu = torch.where(x < 1e-6,
                      2.0 * K_BOLTZ_k * temp[:, None] * freq[None, :]**2 / C_LIGHT_CM_k**2,
                      const * freq[None, :]**3 / torch.expm1(safe))
    return torch.where(torch.isfinite(bnu), bnu, torch.zeros_like(bnu))

print("COULFF, Planck, and partition helpers ready")''')

md(r"""The H-minus free-free table is stored in two halves. We join them, transform to the logarithmic table used by the tabulated calculation, and keep the wavelength-grid logarithm ready for interpolation.""")

code(r'''# H-minus free-free log table.
nthetaff = HMINOP_THETAFF_t.shape[0]
iw = torch.arange(22, dtype=torch.int64, device=EDEV)
it = torch.arange(nthetaff, dtype=torch.int64, device=EDEV)
ffbeg = HMINOP_FFBEG_t.index_select(0, torch.clamp(iw, max=10))[:, it]
ffend = HMINOP_FFEND_t.index_select(0, torch.clamp(iw - 11, min=0))[:, it]
ff_full = torch.where(iw[:, None] < 11, ffbeg, ffend)
FFLOG_t = torch.log(ff_full / HMINOP_THETAFF_t[None, :] * 5040.0 * K_BOLTZ_k)
WFFLOG_t = torch.log(91.134 / HMINOP_WAVEK_t)
print("H-minus free-free log table assembled")''')

md(r"""The H-minus opacity routine is the dominant source. Bound-free is the tabulated photodetachment cross-section times the H-minus Saha population and the stimulated-emission factor; free-free is the two-dimensional table interpolation in wavelength and $\theta=5040/T$.""")

code(r'''def hminus_opacity_torch(sample_frequency_hz, exp_minus_hnu_over_kT, stimulated_emission):
    """Return tabulated H-minus bound-free plus free-free opacity.

    The result is a ``(depth, frequency)`` mass opacity.  Bound-free uses the
    parabolic photodetachment cross-section table and the H-minus Saha abundance.
    Free-free first interpolates the wavelength table for every frequency, then
    linearly interpolates those values in ``theta = 5040/T`` for every depth.
    ``stimulated_emission`` is accepted for interface symmetry; the bound-free
    branch uses the equivalent ``1 - exp_minus_hnu_over_kT`` factor used by the
    tabulated formula.
    """
    temperature_k = Tk
    theta = 5040.0 / temperature_k
    neutral_h_population = xnfph_t[:, 0]
    hminus_population = (
        torch.exp(0.754209 / (temperature_k * KBOLTZ_EV_k))
        / (2.0 * 2.4148e15 * temperature_k * torch.sqrt(temperature_k))
        * neutral_h_population
        * ne_k
    )
    wavelength_nm = C_LIGHT_NM_k / sample_frequency_hz

    freefree_table_vs_theta = torch.exp(
        linter_matrix_torch(WFFLOG_t, FFLOG_t, torch.log(wavelength_nm))
    )  # (nf, ntheta)
    # Temperature interpolation is depth-vectorized: each layer chooses its
    # theta bracket, while all frequency rows are gathered in one tensor.
    theta_index = torch.clamp(
        torch.searchsorted(HMINOP_THETAFF_t, theta, right=True),
        1,
        nthetaff - 1,
    )
    theta_left = HMINOP_THETAFF_t.index_select(0, theta_index - 1)
    theta_right = HMINOP_THETAFF_t.index_select(0, theta_index)
    freefree_left = freefree_table_vs_theta[:, theta_index - 1]
    freefree_right = freefree_table_vs_theta[:, theta_index]
    theta_fraction = (
        (theta - theta_left)
        / torch.where(
            torch.abs(theta_right - theta_left) < 1e-40,
            torch.ones_like(theta_right),
            theta_right - theta_left,
        )
    )[None, :]
    freefree_cross_section = (
        freefree_left + (freefree_right - freefree_left) * theta_fraction
    ).T
    hminus_freefree = (
        freefree_cross_section
        * neutral_h_population[:, None]
        * 2.0
        * ne_k[:, None]
        / rho_k[:, None]
        * 1e-26
    )

    boundfree_cross_section = torch.where(
        sample_frequency_hz > 1.82365e14,
        parabolic_table_torch(HMINOP_WBF_t, HMINOP_BF_t, wavelength_nm),
        torch.zeros_like(sample_frequency_hz),
    )
    hminus_boundfree = (
        boundfree_cross_section[None, :]
        * 1e-18
        * (1.0 - exp_minus_hnu_over_kT)
        * hminus_population[:, None]
        / rho_k[:, None]
    )
    return hminus_boundfree + hminus_freefree''')

md(r"""The hydrogen continuum adds H I bound-free from the Karzas--Latter levels and proton free-free using the charge-1 Coulomb Gaunt factor.""")

code(r'''def hydrogen_opacity_torch(
    sample_frequency_hz,
    exp_minus_hnu_over_kT,
    stimulated_emission,
    hc_over_kT_cm,
    charge1_gaunt,
):
    """Compute H I bound-free and proton free-free continuum opacity.

    The hydrogenic bound-free terms are summed from the Karzas--Latter lookup
    for the explicit levels carried by the table set, with threshold
    masks in wavenumber.  The final addend is H+ free-free using the charge-1
    COULFF table supplied as ``charge1_gaunt``.  All terms return as
    ``(depth, frequency)``.
    """
    temperature_k = Tk
    frequency_row = sample_frequency_hz[None, :]
    wavenumber_cm = frequency_row / C_LIGHT_CM_k
    inverse_frequency_cubed_scale = 2.815e29 / (frequency_row * frequency_row * frequency_row)
    hydrogen_boundfree = inverse_frequency_cubed_scale * 2.0/2.0 / (RYDBERG_CM_k*hc_over_kT_cm[:, None]) * (
        torch.exp(-torch.maximum(torch.full_like(wavenumber_cm, 109250.336), 109678.764 - wavenumber_cm)*hc_over_kT_cm[:, None]) -
        torch.exp(-109678.764*hc_over_kT_cm[:, None])) * stimulated_emission

    # High-n and low-n edge lists are fixed table records; each loop
    # updates a full depth-by-frequency tensor, not one scalar cell at a time.
    for n, thr, wt, e in [(15,487.456,450.0,109191.313),(14,559.579,392.0,109119.188),(13,648.980,338.0,109029.789),(12,761.649,288.0,108917.117),(11,906.426,242.0,108772.336),(10,1096.776,200.0,108581.992),(9,1354.044,162.0,108324.719),(8,1713.713,128.0,107965.051),(7,2238.320,98.0,107440.444)]:  # JUSTIFIED-LOOP: fixed H I high-level bf table.
        cross_section = xkarsas_vec(sample_frequency_hz, 1.0, n, n)[None, :]
        hydrogen_boundfree = hydrogen_boundfree + torch.where(
            wavenumber_cm >= thr,
            cross_section*wt*torch.exp(-e*hc_over_kT_cm[:, None])*stimulated_emission,
            torch.zeros_like(hydrogen_boundfree),
        )

    for n, thr, wt, e in [(6,3046.604,72.0,106632.160),(5,4387.113,50.0,105291.651),(4,6854.871,32.0,102823.893),(3,12186.462,18.0,97492.302),(2,27419.659,8.0,82259.105)]:  # JUSTIFIED-LOOP: fixed H I low-level bf table.
        cross_section = xkarsas_vec(sample_frequency_hz, 1.0, n, n)[None, :]
        hydrogen_boundfree = hydrogen_boundfree + torch.where(
            wavenumber_cm >= thr,
            cross_section*wt*torch.exp(-e*hc_over_kT_cm[:, None])*(1.0 - exp_minus_hnu_over_kT),
            torch.zeros_like(hydrogen_boundfree),
        )

    ground_cross_section = xkarsas_vec(sample_frequency_hz, 1.0, 1, 1)[None, :]
    hydrogen_boundfree = hydrogen_boundfree + torch.where(
        wavenumber_cm >= 109678.764,
        ground_cross_section*2.0*(1.0 - exp_minus_hnu_over_kT),
        torch.zeros_like(hydrogen_boundfree),
    )
    hydrogen_boundfree = hydrogen_boundfree * xnfph_t[:, 0:1] / rho_k[:, None]
    proton_freefree = (
        3.6919e8 / torch.sqrt(temperature_k[:, None])
        * charge1_gaunt.T / frequency_row
        * ne_k[:, None] / frequency_row
        * xnfph_t[:, 1:2] / frequency_row
        * stimulated_emission
        / rho_k[:, None]
    )
    return hydrogen_boundfree + proton_freefree''')

md(r"""## Summing to the absorption and scattering coefficients

The driver below computes the continuum at the selected edge-triplet frequencies. The output arrays are `(depth, sampled_frequency)`: one for true absorption and one for scattering. The final opacity is an explicit sum of the source tensors.""")

md(r"""### The minor absorbers, in one routine

The long tail of continuum sources is small in the solar optical but important for a complete budget. The routine returns two arrays: minor true absorption and minor scattering. Each thresholded term is added only where the photon has enough energy for that process.""")

code(r'''def minor_terms_torch(freq_vec, ehvkt, stim, hckt):
    """Compute the smaller continuum absorbers and minor scattering sources.

    Returns ``(abs_minor, scat_minor)``, both with shape ``(depth, frequency)``.
    This routine collects H2+, He- free-free, C I, Mg I, Al I, Si I, He Rayleigh,
    and H2 Rayleigh/CIA-like scattering terms that are small in the solar
    optical window but part of the tabulated continuum budget.
    """
    temp = Tk
    tkev = temp * KBOLTZ_EV_k
    tlog = torch.log(torch.clamp(temp, min=1e-10))
    xnfph1 = xnfph_t[:, 0]
    xnfph2 = xnfph_t[:, 1]
    f = freq_vec[None, :]
    wno = f / C_LIGHT_CM_k
    abs_minor = torch.zeros((n_layers_k, freq_vec.shape[0]), dtype=EDTYPE, device=EDEV)
    scat_minor = torch.zeros_like(abs_minor)

    # H2+ photodissociation-like contribution, active only below the Lyman edge.
    freqlg = torch.log(f); freq15 = f / 1e15
    fr = -3.0233e3 + (3.7797e2 + (-1.82496e1 + (3.9207e-1 - 3.1672e-3*freqlg)*freqlg)*freqlg)*freqlg
    es = -7.342e-3 + (-2.409e0 + (1.028e0 + (-4.230e-1 + (1.224e-1 - 1.351e-2*freq15)*freq15)*freq15)*freq15)*freq15
    h2p = torch.exp(-es / tkev[:, None] + fr + torch.log(torch.clamp(xnfph1[:, None], min=1e-40))) * 2.0 * xnfph2[:, None] / rho_k[:, None] * stim
    abs_minor = abs_minor + torch.where(f <= 3.28805e15, h2p, torch.zeros_like(h2p))

    ac = 3.397e-01 + (-5.216e14 + 7.039e30/f)/f
    bc = -4.116e03 + (1.067e19 + 8.135e34/f)/f
    cc = 5.081e08 + (-8.724e22 - 5.659e37/f)/f
    abs_minor = abs_minor + (ac*temp[:, None] + bc + cc/temp[:, None]) / 1e15 * ne_k[:, None]/1e15 * xnf_he1_mode11_t[:, None]/1e15 / rho_k[:, None]

    # The following neutral metal edge lists are fixed-size tables;
    # torch.where keeps the thresholding lane-wise across the frequency row.
    c1 = 1e-30 * torch.ones_like(abs_minor)
    c1 = c1 + torch.where(wno >= 22006.370, 2.1e-18*(22006.370/wno)**1.5 * 3.0*torch.exp(-68856.33*hckt[:, None])*stim, torch.zeros_like(abs_minor))
    abs_minor = abs_minor + c1 * xnfpc_t[:, 0:1] / rho_k[:, None]

    mg = 1e-30 * torch.ones_like(abs_minor)
    for thr, c0, p0, g, e in [(13713.986,25e-18,2.7,15.0,47957.034),(13823.223,33.8e-18,2.8,9.0,47847.797),(15267.955,45e-18,2.7,5.0,46403.065),(18167.687,0.43e-18,2.6,1.0,43503.333),(20473.617,2.1e-18,2.6,3.0,41197.043)]:  # JUSTIFIED-LOOP: five fixed Mg I edge records.
        mg = mg + torch.where(wno >= thr, c0*(thr/wno)**p0 * g*torch.exp(-e*hckt[:, None])*stim, torch.zeros_like(mg))
    abs_minor = abs_minor + mg * xnfpmg_t[:, None] / rho_k[:, None]

    al = 1e-30 * torch.ones_like(abs_minor)
    bal2 = torch.exp(-48278.37*hckt)[:, None]
    for thr, c0, p0, g, e in [(8002.467,50e-18,3.0,6.0,40275.903),(9346.231,50e-18,3.0,10.0,38932.139),(10588.957,56.7e-18,1.9,2.0,37689.413),(15318.007,14.5e-18,1.0,6.0,32960.363),(15842.129,47e-18,1.83,10.0,32436.241)]:  # JUSTIFIED-LOOP: five fixed Al I edge records.
        al = al + torch.where(wno >= thr, c0*(thr/wno)**p0 * g*torch.exp(-e*hckt[:, None])*(1.0 - bal2*ehvkt), torch.zeros_like(al))
    abs_minor = abs_minor + al * xnfpal_t[:, None] / rho_k[:, None]

    si = 1e-30 * torch.ones_like(abs_minor)
    si = si + torch.where(wno >= 17777.641, 18e-18*(17777.641/wno)**3 * 15.0*torch.exp(-48161.459*hckt[:, None])*(1.0 - ehvkt), torch.zeros_like(si))
    abs_minor = abs_minor + si * xnfpsi_t[:, None] / rho_k[:, None]

    wave_he = 2.99792458e18 / torch.minimum(freq_vec, torch.full_like(freq_vec, 5.15e15))
    ww = wave_he[None, :]**2
    sig_he = 5.484e-14/(ww*ww)*(1.0 + (2.44e5 + 5.94e10/torch.clamp(ww - 2.90e5, min=1e-10))/ww)**2
    scat_minor = scat_minor + sig_he * xnf_he1_mode11_t[:, None] / rho_k[:, None]

    poly_T = (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 + (1.06206e-18 - 3.08720e-23*temp)*temp)*temp)*temp)*temp)*temp
    uph = hydrogen_partition_torch(temp)
    xnh2 = (xnf_h_t / uph * 2.0)**2 * torch.exp(torch.clamp(4.478/tkev - 4.64584e1 + poly_T - 1.5*tlog, -100, 100)) / rho_k
    wave_h2 = 2.99792458e18 / torch.minimum(freq_vec, torch.full_like(freq_vec, 2.922e15))
    ww2 = wave_h2[None, :]**2
    scat_minor = scat_minor + (8.14e-13 + 1.28e-6/ww2 + 1.61/(ww2*ww2))/(ww2*ww2) * xnh2[:, None]
    return abs_minor, scat_minor''')

md(r"""## The helium continuum and the hot-star term

Helium and hot-star continua are negligible in this solar optical window but part of the complete tabulated budget. We include the He I/He II bound-free/free-free structure and the HOTOP/Si II terms.

One table convention matters only at extremely deep layers: the He II free-free `COULFF` lookup uses a fixed logarithmic frequency, while the explicit $f^{-3}$ factor still uses the current frequency. This is a convention of the tabulated continuum formula, not a new physical source term.""")

code(r'''def helium_opacity_torch(freq_vec, ehvkt, stim, hckt, cff1, cff2):
    """Compute He I and He II bound-free/free-free opacity.

    Returns the pair ``(ahe1, ahe2)`` in ``(depth, frequency)`` order.  The He I
    and He II bound-free pieces are short fixed tables of thresholds and
    coefficients.  Free-free uses the charge-1 and charge-2 COULFF tables
    supplied by the driver; the charge-2 table may carry the fixed logarithmic
    lookup described in the surrounding text.
    """
    temp = Tk
    f = freq_vec[None, :]
    wno = f / C_LIGHT_CM_k
    freqlg = torch.log(freq_vec)[None, :]
    freq3 = 2.815e29 / (f*f*f)

    h = freq3*4.0/2.0/(109722.267*hckt[:, None]) * (
        torch.exp(-torch.maximum(torch.full_like(wno, 195262.919), 198310.76 - wno)*hckt[:, None]) -
        torch.exp(-198310.76*hckt[:, None])) * stim

    # He I excited-level edge tables.  These loops are fixed physics records;
    # each iteration adds the contribution for every layer and frequency.
    for thr, g, e, bi in [(4368.190,3.0,193942.57,28),(4388.260,9.0,193922.5,27),(4388.260,27.0,193922.5,26),(4389.390,7.0,193921.37,25),(4389.450,15.0,193921.31,24),(4392.369,5.0,193918.391,23),(4393.515,15.0,193917.245,22),(4509.980,9.0,193800.78,21),(4647.133,1.0,193663.627,20),(4963.671,3.0,193347.089,19),(6817.943,3.0,191492.817,18),(6858.680,7.0,191452.08,17),(6858.960,21.0,191451.80,16),(6864.201,5.0,191446.559,15),(6866.172,15.0,191444.588,14),(7093.620,9.0,191217.14,13),(7370.429,1.0,190940.331,12),(8012.550,3.0,190298.210,11)]:  # JUSTIFIED-LOOP: fixed He I n=5,4 level table.
        x = freq3 / (3125.0 if bi >= 19 else 1024.0)
        h = h + torch.where(wno >= thr, x*g*torch.exp(-e*hckt[:, None])*(1.0 - ehvkt), torch.zeros_like(h))

    for thr, cf0, cf1, g, e in [(12101.289,58.81,-2.89,3.0,186209.471),(12205.695,85.20,-3.69,5.0,186105.065),(12209.106,85.20,-3.69,15.0,186101.654),(12746.066,49.30,-2.60,9.0,185564.694),(13445.824,23.85,-1.86,1.0,184864.936),(15073.868,12.69,-1.54,3.0,183236.892),(27175.760,81.35,-3.5,3.0,171135.000),(29223.753,61.21,-2.9,9.0,169087.007),(32033.214,26.83,-1.91,1.0,166277.546)]:  # JUSTIFIED-LOOP: fixed He I n=3,2 level table.
        x = torch.exp(cf0 + cf1*freqlg)
        h = h + torch.where(wno >= thr, x*g*torch.exp(-e*hckt[:, None])*(1.0 - ehvkt), torch.zeros_like(h))

    h = h + torch.where(wno >= 38454.691, torch.exp(-390.026 + (21.035 - 0.318*freqlg)*freqlg)*3.0*torch.exp(-159856.069*hckt[:, None])*(1.0 - ehvkt), torch.zeros_like(h))
    h = h + torch.where(wno >= 198310.760, torch.exp(33.32 - 2.0*freqlg)*(1.0 - ehvkt), torch.zeros_like(h))
    h = h * xnf_he1_mode11_t[:, None] / rho_k[:, None]
    ahe1 = h + 3.619e8/torch.sqrt(temp[:, None]) * cff1.T / f * ne_k[:, None] / f * xnf_he2_mode11_t[:, None] / f * stim / rho_k[:, None]

    h2 = freq3*16.0*2.0/2.0/(438889.068*hckt[:, None]) * (
        torch.exp(-torch.maximum(torch.full_like(wno, 434519.959), 438908.85 - wno)*hckt[:, None]) -
        torch.exp(-438908.85*hckt[:, None])) * stim * xnf_he2_mode11_t[:, None] / rho_k[:, None]

    # He II edge records use hydrogenic charge scaling plus fitted polynomial
    # corrections near selected thresholds.
    for thr, wt, e, div in [(5418.390,162.0,433490.46,59049.0),(6857.660,128.0,432051.19,32768.0),(8956.950,98.0,429951.90,16807.0)]:  # JUSTIFIED-LOOP: three fixed He II levels.
        h2 = h2 + torch.where(wno >= thr, freq3*16.0/div*wt*torch.exp(-e*hckt[:, None])*stim*xnf_he2_mode11_t[:, None]/rho_k[:, None], torch.zeros_like(h2))

    for thr, wt, e, div, p0, p1, p2 in [(12191.437,72.0,426717.413,7776.0,1.0986,-2.704e13,1.229e27),(17555.715,50.0,421353.135,3125.0,1.102,-3.909e13,2.371e27),(27430.925,32.0,411477.925,1024.0,1.101,-5.765e13,4.593e27),(48766.491,18.0,390142.359,243.0,1.101,-9.863e13,1.035e28),(109726.529,8.0,329182.321,32.0,1.105,-2.375e14,4.077e28),(438908.850,2.0,0.0,1.0,0.9916,2.719e13,-2.268e30)]:  # JUSTIFIED-LOOP: six fixed He II polynomial records.
        x = freq3*16.0/div*(p0 + (p1 + p2/f)/f)
        fac = (1.0 - ehvkt) if e == 0.0 else torch.exp(-e*hckt[:, None])*(1.0 - ehvkt)
        h2 = h2 + torch.where(wno >= thr, x*wt*fac*xnf_he2_mode11_t[:, None]/rho_k[:, None], torch.zeros_like(h2))

    ahe2 = h2 + 3.6919e8*4.0/torch.sqrt(temp[:, None]) * cff2.T / f * ne_k[:, None] / f * xnf_he3_mode11_t[:, None] / f * stim / rho_k[:, None]
    return ahe1, ahe2''')

md(r"""The Si II Peach-table term is a two-axis interpolation: first in frequency, then in temperature. The frequency bracket is common to all layers, while the temperature bracket is a depth vector.""")

code(r'''def si2op_torch(freq_vec, tlog):
    """Interpolate the Si II Peach opacity table over frequency and temperature.

    The frequency bracket is chosen for every sampled frequency, the temperature
    bracket for every depth, and the final result is transposed into
    ``(depth, frequency)``.  The factor of six is the statistical-weight scale
    used by the tabulated routine.
    """
    nt = torch.clamp((Tk/2000.0).to(torch.int64) - 4, 1, 5)
    dt = (tlog - SI2_TLG_t.index_select(0, nt - 1)) / (SI2_TLG_t.index_select(0, nt) - SI2_TLG_t.index_select(0, nt - 1))

    # First interpolate along the frequency axis of the Peach table.
    gt = (freq_vec[:, None] > SI2_FREQSI_t[None, :]).to(torch.int64)
    n_raw = torch.argmax(gt, dim=1) + 1
    n_raw = torch.where(torch.any(gt.bool(), dim=1), n_raw, torch.full_like(n_raw, 8))
    n_next = torch.clamp(n_raw, max=8)
    d = (torch.log(freq_vec) - SI2_FLOG_t.index_select(0, n_raw - 1)) / (
        SI2_FLOG_t.index_select(0, n_next) - SI2_FLOG_t.index_select(0, n_raw - 1)
    )

    n2 = torch.where(n_raw > 2, 2*n_raw - 2, n_raw)
    n2 = torch.clamp(n2, max=13)
    x_hi = SI2_PEACH_t.index_select(0, n2)
    x_lo = SI2_PEACH_t.index_select(0, torch.clamp(n2 - 1, min=0))
    x = torch.where((n2 > 0)[:, None], x_hi*d[:, None] + x_lo*(1.0 - d[:, None]), x_hi)

    # Then gather the two temperature rows needed by each atmospheric layer.
    row0 = torch.gather(x, 1, (nt - 1)[None, :].expand(freq_vec.shape[0], -1))
    row1 = torch.gather(x, 1, nt[None, :].expand(freq_vec.shape[0], -1))
    return (torch.exp(row0*(1.0 - dt[None, :]) + row1*dt[None, :]) * 6.0).T''')

md(r"""The HOTOP term combines multi-charge free-free with a table of bound-free transitions. The tensor shapes are explicit: transitions $\times$ depth $\times$ frequency for the bound-free addends, reduced over the transition axis.""")

code(r'''def hot_and_si2_torch(freq_vec, stim, tkev, tlog, cff_tabs):
    """Compute HOTOP free/free-bound opacity plus the Si II Peach term.

    ``cff_tabs`` supplies charge-indexed COULFF tensors from the driver.  HOTOP
    starts with multi-charge free-free opacity, then adds bound-free transition
    records where the transition cross-section is significant relative to the
    free-free background.  The returned pair is ``(ahot, aluke)`` in
    ``(depth, frequency)`` order.
    """
    free = torch.zeros((n_layers_k, freq_vec.shape[0]), dtype=EDTYPE, device=EDEV)
    for q in range(1, 6):  # JUSTIFIED-LOOP: five fixed ionic charges; each is a full tensor add.
        free = free + cff_tabs[q].T * xnf_sumqq_t[:, q-1:q]

    ahot = free * (3.6919e8 / freq_vec[None, :]**3) * (ne_k[:, None] / torch.sqrt(Tk[:, None]))

    tr = HOTOP_TRANSITIONS_t
    f0 = tr[:, 0]; xs0 = tr[:, 1]; al0 = tr[:, 2]; pw0 = tr[:, 3]; mu0 = tr[:, 4]; e0 = tr[:, 5]
    hid = torch.clamp(tr[:, 6].to(torch.int64) - 1, 0, 20)

    # Broadcast transitions x depth x frequency, then reduce over transitions.
    ratio = f0[:, None] / freq_vec[None, :]
    active = freq_vec[None, :] >= f0[:, None]
    xsect = xs0[:, None] * (al0[:, None] + ratio - al0[:, None]*ratio) * torch.sqrt(torch.pow(ratio, pw0[:, None]))
    pop_hot = hotop_xnfp_t.index_select(1, hid).permute(1, 0)[:, :, None]   # (ntrans, layer, 1)
    xx = xsect[:, None, :] * pop_hot * mu0[:, None, None]                  # (ntrans, layer, nfreq)
    add = xx * torch.exp(-e0[:, None, None] / torch.clamp(tkev[None, :, None], min=1e-30))
    free_ref = ahot[None, :, :] / 100.0                                    # (1, layer, nfreq)
    ahot = ahot + torch.sum(torch.where((xx > free_ref) & active[:, None, :], add, torch.zeros_like(add)), dim=0)
    ahot = ahot * stim / rho_k[:, None]

    aluke = si2op_torch(freq_vec, tlog) * pop_t[:, 1, 13:14] * stim / rho_k[:, None]
    return ahot, aluke''')

md(r"""Rayleigh and Thomson are the scattering part of the continuum. In this optical window the Gavrila hydrogen Rayleigh table is the active branch; the code below keeps the branch as a branchless tensor selection.""")

code(r'''def rayleigh_G_torch(freq_vec):
    FREQ_LYMAN = 3.288051e15
    FREQ_STEP = 3.288051e13
    i = torch.clamp((freq_vec / FREQ_STEP).to(torch.int64) + 1, 1, 74)
    im = torch.clamp(i - 2, 0, HRAYOP_GAVRILAM_t.shape[0] - 1)
    ip = torch.clamp(i - 1, 0, HRAYOP_GAVRILAM_t.shape[0] - 1)
    lin = HRAYOP_GAVRILAM_t.index_select(0, im) + (
        HRAYOP_GAVRILAM_t.index_select(0, ip) - HRAYOP_GAVRILAM_t.index_select(0, im)
    ) / FREQ_STEP * (freq_vec - (i.to(EDTYPE) - 1.0) * FREQ_STEP)
    low = HRAYOP_GAVRILAM_t[0] * (freq_vec / FREQ_STEP)**2
    return torch.where(freq_vec < FREQ_LYMAN * 0.01, low, lin)

def scattering_opacity_torch(freq_vec):
    uph = hydrogen_partition_torch(Tk)
    g = rayleigh_G_torch(freq_vec)
    sigh = 6.65e-25 * g[None, :]**2 * (xnf_h_t / uph)[:, None] * 2.0 / rho_k[:, None]
    sigel = 0.6653e-24 * ne_k[:, None] / rho_k[:, None] * torch.ones((1, freq_vec.shape[0]), dtype=EDTYPE, device=EDEV)
    return sigh, sigel''')

md(r"""## Assembling the tabulated continuum

The driver computes all frequency-dependent and temperature-dependent invariants once, batches the Coulomb Gaunt factor over all selected sample frequencies, and then adds every opacity source. The returned budget dictionary is for inspection; the arrays used downstream are the two genuine sums `acont` and `sigmac`.""")

code(r'''def compute_kapp_at_freqs_torch(sample_frequency_hz):
    """Evaluate the tabulated continuum at selected sample frequencies.

    This is the driver that prepares shared thermal factors, computes all
    charge-specific COULFF tables once, calls every source routine, and returns
    the two physical sums: true absorption ``acont`` and scattering ``sigmac``.
    The third return value is a named source budget for interpretation and plots.
    """
    temperature_k = Tk
    hc_over_kT_cm = H_PLANCK_k / (K_BOLTZ_k * temperature_k) * C_LIGHT_CM_k
    temperature_ev = temperature_k * KBOLTZ_EV_k
    log_temperature = torch.log(torch.clamp(temperature_k, min=1e-10))
    exp_minus_hnu_over_kT = torch.exp(
        -H_PLANCK_k * sample_frequency_hz[None, :] / (K_BOLTZ_k * temperature_k[:, None])
    )
    stimulated_emission = 1.0 - exp_minus_hnu_over_kT

    # Build the small set of charge-specific Gaunt-factor tables once and share
    # them across hydrogen, helium, and HOTOP.
    gaunt_by_charge = {q: coulff_table_torch(q, sample_frequency_hz, temperature_k, log_temperature) for q in range(1, 6)}  # JUSTIFIED-LOOP: five fixed ionic charges.
    # He II table convention: use a fixed logarithmic lookup frequency for this Gaunt table.
    helium2_charge2_gaunt = coulff_table_torch(2, sample_frequency_hz, temperature_k, log_temperature, torch.log(freqset_t[-1]))

    hminus_absorption = hminus_opacity_torch(sample_frequency_hz, exp_minus_hnu_over_kT, stimulated_emission)
    hydrogen_absorption = hydrogen_opacity_torch(
        sample_frequency_hz,
        exp_minus_hnu_over_kT,
        stimulated_emission,
        hc_over_kT_cm,
        gaunt_by_charge[1],
    )
    minor_absorption, minor_scattering = minor_terms_torch(sample_frequency_hz, exp_minus_hnu_over_kT, stimulated_emission, hc_over_kT_cm)
    helium1_absorption, helium2_absorption = helium_opacity_torch(
        sample_frequency_hz,
        exp_minus_hnu_over_kT,
        stimulated_emission,
        hc_over_kT_cm,
        gaunt_by_charge[1],
        helium2_charge2_gaunt,
    )
    hot_absorption, si2_absorption = hot_and_si2_torch(sample_frequency_hz, stimulated_emission, temperature_ev, log_temperature, gaunt_by_charge)
    rayleigh_scattering, thomson_scattering = scattering_opacity_torch(sample_frequency_hz)

    continuum_absorption = (
        hminus_absorption
        + hydrogen_absorption
        + minor_absorption
        + helium1_absorption
        + helium2_absorption
        + hot_absorption
        + si2_absorption
    )
    continuum_scattering = rayleigh_scattering + thomson_scattering + minor_scattering
    source_budget = dict(
        Hminus=hminus_absorption,
        HI=hydrogen_absorption,
        minor=minor_absorption,
        He=helium1_absorption + helium2_absorption,
        hot=hot_absorption + si2_absorption,
        Rayleigh=rayleigh_scattering,
        Thomson=thomson_scattering,
        scat_minor=minor_scattering,
    )
    return continuum_absorption, continuum_scattering, source_budget

absorption_sampled, scattering_sampled, source_budget = compute_kapp_at_freqs_torch(freq_sel_t)
print(f"computed tabulated sources at {int(freq_sel_t.shape[0])} sampled frequencies")''')

code(r'''# Physical sanity: sampled opacity coefficients should be finite and positive before interpolation.
finite_ok = bool(torch.all(torch.isfinite(absorption_sampled)) and torch.all(torch.isfinite(scattering_sampled)))
amin = float(torch.min(absorption_sampled)); smin = float(torch.min(scattering_sampled))
print(f"sampled coefficients: finite={finite_ok}, min(abs)={amin:.3e}, min(scat)={smin:.3e}")''')

md(r"""## Reading the budget: who absorbs and who scatters

At a representative photospheric layer the tabulated budget should look like the physical story: H-minus dominates the absorption, H I and molecular/metal/helium terms fill in the remaining few percent, and scattering is mostly Rayleigh plus Thomson. The numbers are computed directly from the source tensors above.""")

code(r'''layer_budget = int(torch.argmin(torch.abs(Tk - 6400.0)))
jfreq_budget = 0
a_tot_b = absorption_sampled[layer_budget, jfreq_budget]
s_tot_b = scattering_sampled[layer_budget, jfreq_budget]
print(f"per-source budget at layer {layer_budget}, T={float(Tk[layer_budget]):.0f} K:")
for name in ("Hminus", "HI", "minor", "He", "hot"):  # JUSTIFIED-LOOP: five named reporting terms, no computation over data axes.
    val = source_budget[name][layer_budget, jfreq_budget]
    print(f"  ABS  {name:8s} = {float(val):.4e}  ({float(100*val/a_tot_b):6.2f}%)")
for name in ("Rayleigh", "Thomson", "scat_minor"):  # JUSTIFIED-LOOP: three named reporting terms.
    val = source_budget[name][layer_budget, jfreq_budget]
    print(f"  SCAT {name:8s} = {float(val):.4e}  ({float(100*val/s_tot_b):6.2f}%)")''')

md(r"""## The 3-point Lagrange interpolation to any wavelength

The tabulated continuum stores $\log_{10}\kappa$ at the three sample frequencies of each edge interval. For every synthesis wavelength we build the three Lagrange basis coefficients from the edge wavelengths and the midpoint, multiply them by the stored logs, and exponentiate back to the opacity. This reconstructs the continuum smoothly between edges while preserving the edge sampling.""")

code(r'''half_edge_t = dev64(A_np["half_edge"])
delta_edge_t = dev64(A_np["delta_edge"])

e = edge_idx_t.to(torch.int64)
wl_l = wledge_t.index_select(0, e)
wl_r = wledge_t.index_select(0, e + 1)
half = half_edge_t.index_select(0, e)
delta = torch.where(torch.abs(delta_edge_t.index_select(0, e)) > 0.0,
                    delta_edge_t.index_select(0, e),
                    torch.full_like(wlk, 1e-20))
c1_lag = (wlk - half) * (wlk - wl_r) / delta
c2_lag = (wl_l - wlk) * (wlk - wl_r) * 2.0 / delta
c3_lag = (wlk - wl_l) * (wlk - half) / delta
basis_lag = torch.stack((c1_lag, c2_lag, c3_lag), dim=0)

triplet_index = torch.searchsorted(used_edges_t, e).to(torch.int64)
abs_trip = torch.log10(torch.clamp(absorption_sampled.reshape(n_layers_k, used_edges_t.shape[0], 3), min=1e-30))
sca_trip = torch.log10(torch.clamp(scattering_sampled.reshape(n_layers_k, used_edges_t.shape[0], 3), min=1e-30))
abs_w_trip = abs_trip.index_select(1, triplet_index)
sca_w_trip = sca_trip.index_select(1, triplet_index)

absorption_tabulated = 10.0 ** torch.sum(abs_w_trip * basis_lag.T[None, :, :], dim=2)
scattering_tabulated = 10.0 ** torch.sum(sca_w_trip * basis_lag.T[None, :, :], dim=2)

print(f"interpolated tabulated continuum to {int(wlk.shape[0])} synthesis wavelengths")''')

md(r"""## Reading the tabulated continuum

The tabulated calculation has now produced absorption and scattering on the synthesis wavelength grid. Before plotting, inspect the ranges and confirm that the arrays are finite and positive. Here the goal is to understand the continuum that has been assembled from the source terms.""")

code(r'''abs_tab_np = to_np(absorption_tabulated)
sca_tab_np = to_np(scattering_tabulated)
finite_tabulated = np.isfinite(abs_tab_np).all() and np.isfinite(sca_tab_np).all()
print(f"tabulated continuum finite: {finite_tabulated}")
print(f"absorption range: {abs_tab_np.min():.3e} -> {abs_tab_np.max():.3e} cm^2/g")
print(f"scattering range: {sca_tab_np.min():.3e} -> {sca_tab_np.max():.3e} cm^2/g")''')

md(r"""## Overlay: analytic and tabulated

A single photospheric layer shows the structure of the calculation. The analytic continuum from the first half captures the H-minus scale and shape. The tabulated continuum carries the additional sources and the edge-triplet reconstruction, so it is the one used for detailed synthesis.""")

code(r'''layer_p = int(np.argmin(np.abs(to_np(Tk) - 6400.0)))
analytic_total_np = to_np(kappa_Hminus + kappa_scat)
la3 = int(np.argmin(np.abs(REF["T"] - float(Tk[layer_p]))))
analytic_p = np.interp(to_np(wlk), REF["wl"], analytic_total_np[la3])

tabulated_total_np = abs_tab_np[layer_p] + sca_tab_np[layer_p]

fig, (ax, axr) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
ax.plot(to_np(wlk), tabulated_total_np, color="C3", lw=1.2, label="tabulated continuum")
ax.plot(to_np(wlk), analytic_p, color="C0", lw=1.0, ls="--", label="analytic fits")
ax.set_yscale("log"); ax.set_ylabel(r"$\kappa_{\rm cont}$  [cm$^2$/g]")
ax.set_title(f"Continuum at T = {float(Tk[layer_p]):.0f} K")
ax.legend(loc="center right", fontsize=9)

ratio = analytic_p / np.maximum(tabulated_total_np, 1e-300)
axr.plot(to_np(wlk), ratio, color="C0", lw=0.9)
axr.axhline(1.0, color="0.5", lw=0.8, ls="--")
axr.set_xlabel("wavelength  [nm]"); axr.set_ylabel("analytic / tabulated")
axr.set_ylim(0.75, 1.25)
fig.tight_layout(); plt.show()''')

md(r"""## Synthesis: what you built

You turned the equation of state into a continuum opacity twice. First you built the physical model: the visible continuum of a cool star is carried mainly by H-minus, whose abundance follows a Saha balance with the small detachment energy $\chi=0.754\ \mathrm{eV}$, and whose bound-free and free-free absorption can be represented by the John (1988) analytic fits. Rayleigh and Thomson scattering then add a small non-thermalizing floor.

Then you built the tabulated continuum. Each source was evaluated from its own table or fixed cross-section rule, sampled on an edge-triplet frequency grid, and reconstructed by the 3-point Lagrange interpolation. The array structure is the same throughout: populations are depth columns, frequency quantities are wavelength or frequency rows, and every physical source fills the whole depth--frequency plane.

The result is a smooth continuum opacity, separated into true absorption and scattering, with a source budget that explains why H-minus dominates the solar optical continuum.""")

md(r"""## Summary

- Continuous extinction separates into **true absorption** and **scattering**; only absorption thermalizes photons and carries the stimulated-emission correction.
- In cool photospheres, **H-minus** dominates the visible continuous absorption because its abundance scales with both neutral hydrogen and the electron density supplied by metals.
- The analytic John (1988) H-minus fits plus Rayleigh/Thomson scattering explain the dominant solar optical continuum.
- The tabulated continuum replaces the analytic fits with tables: H-minus bound-free/free-free, H I Karzas--Latter bound-free, Coulomb free-free Gaunt factors, Gavrila Rayleigh scattering, helium continua, and minor molecular/metal absorbers.
- The edge-triplet convention samples three frequencies per continuum edge interval and reconstructs $\log_{10}\kappa$ with a fixed 3-point Lagrange parabola.
- The calculation is organized over depth and sampled frequency; the only loops are over short fixed physics tables, never over individual layers or wavelengths.""")

md(r"""## Practice exercises

**1. The H-minus threshold.** Plot the analytic H-minus bound-free cross-section from $0.2$ to $1.7\,\mu{\rm m}$. Where does it peak, and why must it vanish beyond $1.6419\,\mu{\rm m}$?

**2. Metal-poor continuum.** Reduce the electron density in the analytic H-minus calculation by a factor of ten. How does the continuum opacity shift, and which atmospheric layers would become visible?

**3. Source budget.** Use the source-budget printout to estimate how much opacity remains if only H-minus absorption is kept. Which terms supply the missing opacity in this wavelength window?

**4. Interpolation test.** Replace the 3-point Lagrange reconstruction with a two-point linear interpolation in $\log_{10}\kappa$. How large is the resulting error across the 500--510 nm window?

**5. Table decisions.** Identify one place where the calculation makes a discrete table decision and one place where it evaluates a smooth formula. Why is the discrete decision more sensitive to tiny rounding changes?""")

md(r"""## Further reading

- **John, T. L. (1988), A&A, 193, 189.** Continuous absorption by H-minus and the analytic fits used for the pedagogical model.
- **Kurucz, R. L. (1970), SAO Special Report 309.** The ATLAS continuous-opacity routines and tables.
- **Gavrila, M. (1967), Phys. Rev., 163, 147.** The Rayleigh-scattering polarizability factor for atomic hydrogen.
- **Karzas, W. J. & Latter, R. (1961), ApJS, 6, 167.** Hydrogenic bound-free and free-free Gaunt factors.
- **Wildt, R. (1939), ApJ, 90, 611.** The identification of H-minus as the key stellar photospheric opacity source.
- **Gray, D. F. (2005), *The Observation and Analysis of Stellar Photospheres*.** A clear textbook treatment of continuous opacity.""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT.relative_to(BOOK)} ({len(cells)} cells)")
