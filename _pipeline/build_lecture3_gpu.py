#!/usr/bin/env python
"""Assemble content/Lecture3.ipynb (unexecuted) — the GPU EDITION. Execute + render via build.py.

Lecture 3 (GPU) — Continuous Opacity, ported to clean depth-AND-wavelength-batched torch/MPS.
The analytic continuous opacity of a cool star: the H- ion abundance from a Saha balance, its
bound-free and free-free cross-sections (John 1988), Rayleigh scattering off neutral hydrogen and
Thomson scattering off free electrons — every source a single tensor expression broadcast over the
(80 depth) x (200 wavelength) grid, on the GPU (MPS/CUDA if present, else CPU/fp64). The lecture
ends with TWO comparison cells: against the production reference (the physics-level ~few-percent
fidelity of the analytic model) and against the GPU's OWN numpy fp64 twin (the fp32 float floor,
proving the vectorization is bit-correct).

This is the GPU companion to the NumPy edition's Lecture 3, Part A (the analytic continuum). The
exact tabulated KAPP engine (Part B of the NumPy edition) is the production path; this edition
ports the analytic continuum the reference L3.npz validates, fully vectorized. The notebook imports
neither kgpu nor pykurucz.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture3.ipynb"

cells = []
def md(src): cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

md(r"""# Lecture 3 — Continuous Opacity *(GPU Edition)*

*Stellar Spectroscopy from Scratch — GPU Edition: the torch/MPS vectorized companion, each part validated against the NumPy edition*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*This is the **GPU edition** of Lecture 3. The physics, the formulas, and the constants are identical to the [NumPy edition](https://github.com/tingyuansen/stellar-spectroscopy-from-scratch); the continuous opacity is rebuilt in clean **`torch`** that runs on the GPU (Apple **MPS** or **CUDA**, with a CPU fallback in fp64). The crucial new axis here is **wavelength**: where Lecture 2 batched over the 80 atmospheric depths, the continuum is a function of depth **and** wavelength, so every opacity source is a single tensor expression broadcast over the full $(80 \times 200)$ grid — no Python loop over depths or wavelengths. The lecture ends with two comparison cells: one against the production reference `L3.npz` (the analytic model's honest few-percent physics fidelity), and one against the GPU's **own NumPy fp64 twin** (the fp32 float floor, which proves the vectorization is bit-correct). The notebook imports neither `kgpu` nor pykurucz.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Distinguish **true absorption** from **scattering**, and write the continuous extinction coefficient per gram.
- Solve the **Saha balance** for the negative hydrogen ion H$^-$ and explain why it dominates the cool-star continuum.
- Evaluate the **H$^-$ bound-free and free-free** opacity (John 1988) and add **Rayleigh** and **Thomson** scattering, as tensor expressions broadcast over depth **and** wavelength on the GPU.
- Eliminate the data-dependent threshold branch **branchlessly** — clamp the base before a fractional power — so the GPU never produces a NaN that contaminates an in-range pixel.
- **Validate** the GPU continuum twice: against the production reference (the analytic physics, a few percent) and against the GPU's own fp64 twin (the fp32 float floor), the independent per-part check.""")

md(r"""## Introduction

With the equation of state of Lecture 2 in hand — the electron density and the per-ion populations — we can finally compute an opacity. Opacity comes in two flavours, and the distinction matters for the radiative transfer later. **True absorption** couples the photon to the thermal energy reservoir of the gas; in LTE its emissivity follows the Planck function. **Scattering** merely redirects a photon without exchanging energy, so its source function depends on the radiation field itself. This lecture builds the **continuous** opacity: the smooth, slowly-varying background that sets the overall brightness of the star and the floor from which the sharp spectral lines are carved.

The physics is exactly that of the NumPy edition's Lecture 3 — the analytic continuum: the H$^-$ ion dominates the cool-star continuum, so we solve its Saha abundance, evaluate its bound-free and free-free cross-sections with the standard fits of John (1988), and add Rayleigh and Thomson scattering. Against the production reference this analytic model is accurate to **a few percent** — the right level to understand *why* the continuum looks as it does (the NumPy edition's second half then closes that gap to the bit with the tabulated KAPP engine; that exact engine is the production path, ported in the `kgpu` continuum module).

What changes here is the *machine*. The continuum is a function of two axes — depth and wavelength — so we make **both** the batch: a single `torch` expression evaluates an opacity source at all $80 \times 200$ grid points at once, with the depth-dependent populations broadcast against the wavelength-dependent cross-sections. There is one genuinely GPU-specific subtlety — a data-dependent threshold in the H$^-$ bound-free cross-section — which we handle branchlessly, on the advice of two code critics. At the end we validate against both the reference and the GPU's own fp64 twin.""")

md(r"""**Setup — the device and the precision budget.** As in Lecture 2 we pick the compute device once: **MPS** on Apple Silicon, **CUDA** on an NVIDIA box, else **CPU**. MPS and CUDA have no fp64, so the GPU working dtype is **fp32** and the parity bar is the documented float floor; on CPU we use **fp64** and recover machine precision. We load the reference continuum grid `L3.npz` — the same one the NumPy edition validates against — which carries the wavelength and temperature grids, the populations from the equation of state, the depth scale, and the gold-standard continuum (`absorption`, `scattering`).""")

code(r'''import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt

# pick the compute device ONCE; MPS (Apple) -> CUDA -> CPU. MPS/CUDA have no fp64,
# so the GPU working dtype is fp32 (parity bar = the documented float floor);
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

plt.rcParams.update({
    "figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
})

REF = np.load(pathlib.Path("..") / "reference" / "L3.npz")''')

md(r"""Unpack the grids and populations, reshaping the depth-dependent quantities to a column so they broadcast over the wavelength axis. The wavelength grid is a row. Every opacity below is then a $(\text{nd}, \text{nw})$ tensor formed by broadcasting a depth column against a wavelength row — the GPU evaluates all 16 000 grid points in one shot, no loop on either axis.""")

code(r'''# the grids, reshaped so depth (column) broadcasts against wavelength (row)
wl   = dev(REF["wl"])               # (nw,) [nm]   -- wavelength grid (a row when [None,:])
T    = dev(REF["T"])[:, None]       # (nd,1) [K]
n_e  = dev(REF["n_e"])[:, None]     # (nd,1) [cm^-3]
rho  = dev(REF["rho"])[:, None]     # (nd,1) [g cm^-3]
nHI  = dev(REF["nHI"])[:, None]     # (nd,1) neutral H number density [cm^-3]
nw   = wl.shape[0]

# Rosseland optical depth and column mass (host NumPy — only used for plotting / formation depth)
tau, rhox = REF["tau"], REF["rhox"]

# physical constants, CGS: Planck, c, Boltzmann
H, C, K = 6.62607015e-27, 2.99792458e10, 1.380649e-16
# eV-per-K conversion; Saha (2pi m_e k/h^2)^3/2 prefactor
KEV, SAHA = 1.0/11604.5, 2.4148e15

# frequency [Hz] at each wavelength (a row, broadcasts over depth)
nu = C / (wl[None, :] * 1e-7)

print(f"continuum grid: {REF['absorption'].shape[0]} layers x {nw} wavelengths, "
      f"{float(wl[0]):.0f}-{float(wl[-1]):.0f} nm  (all opacities are (nd, nw) tensors on {DEVICE.type})")''')

md(r"""## The stimulated-emission factor

Every true-absorption coefficient carries a correction we met in Lecture 1: the radiation field stimulates the inverse, emitting transition alongside the absorbing one, so the net absorption is multiplied by

$$
\big(1 - e^{-h\nu/kT}\big),
$$

near $1$ in the blue, dropping toward the infrared where $h\nu \lesssim kT$. Scattering, which exchanges no energy with the gas, carries no such factor. On the GPU this is one broadcast expression: the depth column $T$ against the wavelength row $\nu$ gives the full $(\text{nd}, \text{nw})$ factor at once.""")

code(r'''# stimulated-emission correction, (nd, nw) — depth column T broadcast against wavelength row nu
stim = 1.0 - torch.exp(-H * nu / (K * T))
print(f"stimulated-emission factor at 505 nm: "
      f"{float(stim[:,100].min()):.3f} (hot, deep) .. {float(stim[:,100].max()):.3f} (cool)")''')

md(r"""## How much H$^-$ is there? The Saha balance

In a cool star the dominant continuous absorber is the **negative hydrogen ion, H$^-$** — a neutral hydrogen atom holding a second, weakly bound electron (binding energy $\chi = 0.754\ \mathrm{eV}$). It is fragile and rare, but neutral hydrogen is so abundant and the metal-supplied free electrons (Lecture 2) so available that in solar-type photospheres H$^-$ swamps every other optical continuous opacity. Its abundance follows a Saha balance exactly like Lecture 2's, with the tiny detachment energy in place of an ionization potential:

$$
n(\mathrm{H}^-) = \frac{n(\mathrm{H\,I})\,n_e}{4\,(2.4148\times10^{15})\,T^{3/2}}\;e^{+\chi/kT}.
$$

The positive exponent says H$^-$ is *favoured* at low temperature, and the explicit $n_e$ is why H$^-$ opacity tracks the electron density — and hence the metal abundance — that Lecture 2 worked out. We derive this density once, on the GPU; both opacity channels reuse it.

![The H$^-$ ion — a hydrogen atom holding a second, weakly bound electron (0.754 eV) — is the dominant continuous absorber in cool stars; a photon detaches the electron, and its abundance tracks the electron density.](resources/figures/s3_hminus.png)""")

code(r'''chi_Hminus = 0.754   # H- electron binding energy [eV]

# the Saha balance, solved for the H- density [cm^-3] — a (nd,1) depth column
n_Hminus = nHI * n_e * torch.exp(chi_Hminus / (KEV * T)) / (4.0 * SAHA * T**1.5)

jp = int(np.argmin(np.abs(tau - 2/3)))
print(f"n(H-)/n(H I) at the photosphere: {float((n_Hminus/nHI)[jp,0]):.2e}  "
      f"(about two H- per billion H atoms)")''')

md(r"""## H$^-$ bound-free and free-free (John 1988), and the branchless threshold

H$^-$ absorbs by two channels. **Bound-free** (photodetachment) ejects the bound electron; it has a threshold at $\lambda_0 = 1.6419\ \mathrm{\mu m}$ and peaks in the red. **Free-free** is absorption by a passing electron in the field of a neutral H atom; it has no threshold and rises into the infrared. We use the analytic fits of **John (1988)**.

The bound-free cross-section, with $\lambda$ in microns and $f \equiv 1/\lambda - 1/\lambda_0$, is $\sigma_{\rm bf} = 10^{-18}\,\lambda^3\,f^{3/2}\sum_{n=0}^{5} C_n\, f^{n/2}$, *zero past the threshold* ($\lambda \geq \lambda_0$, where $f \leq 0$). This threshold is the one genuinely GPU-specific subtlety in the lecture: $f$ goes **negative** past threshold, and a fractional power of a negative number is `NaN` — and `torch.where(mask, f**1.5*..., 0)` does **not** help, because `torch` evaluates *both* branches, so the NaN is computed before the mask is applied and can poison neighbouring values. The fix, which two code critics (GPT-5.5 and Gemini-3.1-pro) independently recommended, is to **clamp the base to $\geq 0$ before the power**: with $x = \sqrt{\max(0,\,f)}$, the cross-section becomes $\sigma_{\rm bf} = 10^{-18}\,\lambda^3\,x^3\sum_n C_n\,x^n$, a clean **Horner** polynomial in $x$ that is *exactly* zero past threshold (because $x = 0$ there) with no `NaN` and no `where` at all. This is both branchless and faster — fewer kernel launches — and we use the algebraically stable form $f = (\lambda_0 - \lambda)/(\lambda_0\lambda)$ to avoid catastrophic cancellation near the edge.""")

code(r'''# H- bound-free (John 1988) — the critics' clamp-then-Horner pattern (branchless, NaN-free).
lam_um = wl[None, :] * 1e-3          # (1, nw) [um]
lam0 = 1.6419                        # threshold [um] (0.754 eV)

# stable form of f = 1/lam - 1/lam0 (avoids fp32 cancellation near the edge), then clamp >= 0
f = torch.clamp((lam0 - lam_um) / (lam0 * lam_um), min=0.0)
x = torch.sqrt(f)                    # x = sqrt(max(0,f)); EXACTLY 0 where lam >= lam0 -> no NaN

# sum_n C_n f^(n/2) = sum_n C_n x^n  (Horner); sigma = 1e-18 lam^3 x^3 * poly  (x^3 = f^1.5)
C_bf = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982]
poly = C_bf[5]
for c in reversed(C_bf[:5]):
    poly = poly * x + c
sigma_bf = 1e-18 * lam_um**3 * x**3 * poly       # (1, nw) [cm^2]; zero past threshold by construction

# opacity per gram, with the stimulated-emission factor [cm^2/g] — broadcasts to (nd, nw)
kappa_bf = n_Hminus * sigma_bf * stim / rho''')

md(r"""The free-free coefficient is John's second polynomial: a double power series in $\theta = 5040/T$ and in inverse wavelength, returning absorption per neutral H atom per unit electron pressure $P_e = n_e kT$ (the stimulated-emission factor is already folded into the fit). We evaluate the inverse-wavelength terms with explicit reciprocal powers (the critics' efficiency note) and sum the five $\theta$-orders — the whole thing a $(\text{nd}, \text{nw})$ tensor: $\theta$ is a depth column, the wavelength terms a row.""")

code(r'''# H- free-free (John 1988, lambda > 0.3645 um branch), per H I atom per unit P_e.
A=[0,2483.346,-3449.889,2200.040,-696.271,88.283]; B=[0,285.827,-1158.382,2427.719,-1841.400,444.517]
Cc=[0,-2054.291,8746.523,-13651.105,8624.970,-1863.864]; D=[0,2827.776,-11485.632,16755.524,-10051.530,2095.288]
E=[0,-1341.537,5303.609,-7510.494,4400.067,-901.788]; F=[0,208.952,-812.939,1132.738,-655.020,132.985]

theta = 5040.0 / T                   # (nd,1) John temperature variable (depth column)
inv = 1.0 / lam_um                   # (1,nw) inverse wavelength (row)
kff = torch.zeros_like(stim)         # (nd, nw)
for n in range(1, 6):                # the loop is over the 5 polynomial ORDERS, not depth or wl
    term = A[n]*lam_um**2 + B[n] + Cc[n]*inv + D[n]*inv**2 + E[n]*inv**3 + F[n]*inv**4
    kff = kff + theta**((n+1)/2.0) * term

P_e = n_e * K * T                    # electron pressure [dyn cm^-2] (depth column)
kappa_ff = 1e-29 * kff * P_e * nHI / rho         # [cm^2/g]
kappa_Hminus = kappa_bf + kappa_ff               # total analytic H- absorption (nd, nw)''')

md(r"""**Benchmark against the production reference** — over the layers where the optical spectrum forms. H$^-$ dominates there; far deeper, hydrogen and metal bound-free edges take over from it, and this analytic model does not carry them (the exact engine does). This is the *physics-fidelity* check: the analytic model vs the production continuum.""")

code(r'''def to_np(t):
    """GPU tensor -> NumPy fp64 (move to CPU FIRST, then cast: MPS has no fp64)."""
    return t.detach().cpu().to(torch.float64).numpy() if torch.is_tensor(t) else np.asarray(t, float)

# H- dominates only where the optical spectrum forms; benchmark over the forming layers.
form = (tau > 1e-3) & (tau < 3.0)
absn = to_np(kappa_Hminus)
rel = np.abs(absn[form] - REF["absorption"][form]) / REF["absorption"][form]
print(f"continuum absorption (H-), spectrum-forming layers (vs PRODUCTION reference):")
print(f"   median|rel diff| = {np.median(rel):.2e}   max = {np.max(rel):.2e}  "
      f"(the analytic model's honest few-percent fidelity)")''')

md(r"""About two to three percent through the spectrum-forming layers — the analytic H$^-$ reproduces the production reference, the bound-free channel carrying most of it at these optical wavelengths. The residual is the detailed photodetachment table the production engine carries; it is *not* GPU round-off, and we confirm that explicitly with the fp64-twin check at the end. Where it matters, H$^-$ alone carries the continuous absorption of the solar photosphere.""")

md(r"""## Scattering: Rayleigh beats Thomson

The textbook reflex is **Thomson** scattering off free electrons ($\sigma_T = 0.6653\times10^{-24}\ \mathrm{cm^2}$). But in a cool photosphere the free electrons are scarce while neutral hydrogen is everywhere, so **Rayleigh** scattering off the bound electrons of neutral H — the same $\lambda^{-4}$ scattering that makes the sky blue — dominates. We use the Dalgarno polarizability fit, with $\lambda$ in ångström, and add Thomson. Both are pure wavelength rows scaled by depth populations; we evaluate the inverse powers with explicit reciprocals (the critics' efficiency note).""")

code(r'''lamA = wl[None, :] * 10.0            # (1, nw) [angstrom]
invA = 1.0 / lamA
inv2 = invA*invA; inv4 = inv2*inv2; inv6 = inv4*inv2; inv8 = inv4*inv4

# Rayleigh off neutral H (Dalgarno) + grey Thomson off electrons — broadcast to (nd, nw)
sigma_Ray = 5.799e-13*inv4 + 1.422e-6*inv6 + 2.784*inv8         # [cm^2] per H I atom
kappa_Ray = sigma_Ray * nHI / rho
kappa_Thomson = 0.6653e-24 * n_e / rho * torch.ones_like(nu)
kappa_scat = kappa_Ray + kappa_Thomson
kappa_total = kappa_Hminus + kappa_scat

scatn = to_np(kappa_scat)
rel = np.abs(scatn - REF["scattering"]) / np.where(REF["scattering"]!=0, np.abs(REF["scattering"]), 1.0)
print(f"scattering (Rayleigh+Thomson) vs reference:  max|rel diff| = {rel.max():.2e}")
print(f"at the photosphere, Rayleigh is {float(kappa_Ray[jp,100]/kappa_Thomson[jp,100]):.0f}x Thomson; "
      f"scattering is {float(100*kappa_scat[jp,100]/kappa_Hminus[jp,100]):.1f}% of the H$^-$ absorption")''')

md(r"""Rayleigh outweighs Thomson several-fold at the $\tau=2/3$ layer (the ratio climbs higher in the cooler layers above, where free electrons grow even scarcer), and scattering as a whole is only about a percent of the H$^-$ absorption — the solar optical continuum is an *absorption* continuum, which is why its source function stays close to the Planck function. In the ultraviolet, where $\lambda^{-4}$ Rayleigh climbs and metal photoionization switches on, the balance shifts; in hot stars, where hydrogen is ionized, Thomson takes over.""")

md(r"""## The total continuum and where it forms

Add absorption and scattering for the total continuous extinction, and convert it to optical depth. Here `rhox` is the inward column mass $m = \int\rho\,dz$, so $d\tau_\lambda = \kappa_\lambda\,dm$; integrating the continuum opacity over the column mass tells us the depth from which the continuum escapes.""")

code(r'''totn = to_np(kappa_total); ref_total = REF["absorption"] + REF["scattering"]
rel = np.abs(totn[form] - ref_total[form]) / ref_total[form]
print(f"total continuum, spectrum-forming layers (vs reference):  "
      f"median|rel diff| = {np.median(rel):.2e}   max = {np.max(rel):.2e}")

# continuum optical depth at 505 nm, integrated over column mass (trapezoid), on host NumPy
k505 = totn[:, 100]
tau_cont = np.zeros_like(rhox)
tau_cont[1:] = np.cumsum(0.5*(k505[1:]+k505[:-1]) * np.diff(rhox))
j23 = int(np.argmin(np.abs(tau_cont - 2/3)))
print(f"the 505 nm continuum reaches tau=2/3 at T = {REF['T'][j23]:.0f} K  "
      f"(log tau_Ross = {np.log10(tau[j23]):.2f})")''')

md(r"""Plot the GPU opacity against depth (absorption vs scattering) and the integrated continuum optical depth that locates the $\tau = 2/3$ surface.""")

code(r'''logtau = np.log10(tau)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))
ax[0].plot(logtau, np.log10(totn[:,100]), color="C0", label="total")
ax[0].plot(logtau, np.log10(to_np(kappa_Hminus)[:,100]), "--", color="C3", label="H$^-$ absorption")
ax[0].plot(logtau, np.log10(scatn[:,100]), ":", color="C2", label="scattering")
ax[0].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[0].set_ylabel(r"$\log_{10}\kappa_{505}$  [cm$^2$/g]")
ax[0].set_title("Continuum opacity vs depth (on the GPU)"); ax[0].legend()

ax[1].plot(logtau, tau_cont, color="C0")
ax[1].axhline(2/3, ls="--", color="0.5", lw=1, label=r"$\tau_{505}=2/3$")
ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$")
ax[1].set_ylabel(r"continuum optical depth $\tau_{505}$"); ax[1].set_title("Where the continuum forms")
ax[1].legend()
fig.tight_layout(); plt.show()''')

md(r"""The total continuum matches the production reference to about two percent through the spectrum-forming layers — the level at which clean analytic opacity reproduces a production code, the residual being the detailed cross-section tables it carries. That residual is physics, not GPU round-off — which the next cell proves directly.""")

# ── The validation cell — the per-part GPU check ─────────────────────────
md(r"""## The comparison cell — validating the GPU result two ways

This is the per-part check that defines the GPU edition, and the continuum needs **two** comparisons that mean different things:

1. **vs the production reference (`L3.npz`)** — the *physics* check. The analytic H$^-$ + Rayleigh/Thomson model reproduces the production continuum to a few percent through the spectrum-forming layers. This is a property of the *model*, identical to the NumPy edition; it does not get better on the GPU.

2. **vs the GPU's own NumPy fp64 twin** — the *port-correctness* check. We recompute the **exact same formulas** in fp64 NumPy and compare the GPU fp32 result to them. This isolates the single-precision round-off of the GPU vectorization from the analytic model's physics residual, and it must hold to the documented fp32 float floor. As the critics noted, the H$^-$ bound-free threshold is a zero-crossing, so we measure the relative deviation against $\max(|\text{ref}|,\ \text{floor})$ — pure relative error is meaningless where the cross-section is exactly zero.""")

code(r'''def numpy_twin():
    """Recompute the EXACT same continuum formulas in fp64 NumPy — the GPU's own twin."""
    wl_=REF["wl"]; T_=REF["T"][:,None]; n_e_=REF["n_e"][:,None]; rho_=REF["rho"][:,None]; nHI_=REF["nHI"][:,None]
    nu_=C/(wl_[None,:]*1e-7); stim_=1-np.exp(-H*nu_/(K*T_))
    nHm=nHI_*n_e_*np.exp(chi_Hminus/(KEV*T_))/(4.0*SAHA*T_**1.5)
    lu=wl_[None,:]*1e-3; fr=np.clip((lam0-lu)/(lam0*lu),0,None); xx=np.sqrt(fr)
    pp=C_bf[5]
    for c in reversed(C_bf[:5]): pp=pp*xx+c
    sbf=1e-18*lu**3*xx**3*pp; kbf=nHm*sbf*stim_/rho_
    th=5040.0/T_; iv=1.0/lu; kf=np.zeros_like(stim_)
    for n in range(1,6):
        kf=kf+th**((n+1)/2.0)*(A[n]*lu**2+B[n]+Cc[n]*iv+D[n]*iv**2+E[n]*iv**3+F[n]*iv**4)
    Pe=n_e_*K*T_; kff_=1e-29*kf*Pe*nHI_/rho_
    la=wl_[None,:]*10.0; ia=1.0/la; i2=ia*ia;i4=i2*i2;i6=i4*i2;i8=i4*i4
    kray=(5.799e-13*i4+1.422e-6*i6+2.784*i8)*nHI_/rho_; kth=0.6653e-24*n_e_/rho_*np.ones_like(nu_)
    return kbf+kff_, kray+kth

def floor_rel(name, got, ref, floor):
    """Max relative deviation against max(|ref|, floor) — the absolute floor protects zero-crossings."""
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), floor)
    print(f"{name:36s} max|rel| = {rel.max():.2e}   median = {np.median(rel):.2e}")
    return float(rel.max())

print(f"Validating the GPU continuum against L3.npz and the GPU's own fp64 twin")
print(f"  device = {DEVICE.type}   dtype = {str(DTYPE).split('.')[-1]}\n")

print("-- (1) vs PRODUCTION reference L3.npz (the analytic model's physics fidelity) --")
denom_a = np.where(REF["absorption"]!=0, np.abs(REF["absorption"]), 1.0)
denom_s = np.where(REF["scattering"]!=0, np.abs(REF["scattering"]), 1.0)
phys_abs  = float(np.max(np.abs(absn[form]-REF["absorption"][form])/denom_a[form]))
phys_scat = float(np.max(np.abs(scatn-REF["scattering"])/denom_s))
print(f"  absorption (H-), forming layers       max|rel| = {phys_abs:.2e}   (~few-percent: the analytic fit)")
print(f"  scattering (Rayleigh+Thomson)         max|rel| = {phys_scat:.2e}")

print("\n-- (2) vs the GPU's OWN fp64 twin (the fp32 float floor: GPU port correctness) --")
abs_twin, scat_twin = numpy_twin()
# absolute floor at 1e-6 of the per-array median opacity (protects the bf zero-crossing)
afloor = 1e-6 * np.median(np.abs(abs_twin)); sfloor = 1e-6 * np.median(np.abs(scat_twin))
fa = floor_rel("absorption  GPU vs fp64-twin", absn,  abs_twin,  afloor)
fs = floor_rel("scattering  GPU vs fp64-twin", scatn, scat_twin, sfloor)

max_floor = max(fa, fs)
floor = 5e-5 if DTYPE == torch.float32 else 1e-10
print(f"\nfp32 float floor (GPU vs its own fp64 twin) = {max_floor:.2e}")
status = "PASS" if max_floor < floor else "CHECK"
print(f"documented float floor = {floor:.1e}   ->   [{status}]")
assert max_floor < floor, f"GPU continuum deviates from its fp64 twin by {max_floor:.2e}, above {floor:.1e}"
print("\nThe GPU continuum matches its fp64 twin to the float floor — the vectorization is bit-correct.")''')

md(r"""**What the two numbers mean.** The GPU continuum reproduces its own fp64 twin to the fp32 float floor — a few $\times10^{-6}$ across the grid (the worst case sits near the H$^-$ bound-free threshold zero-crossing, where the cross-section is vanishing and the absolute floor takes over, exactly as the critics anticipated). That residual is single-precision round-off, *not* a physics difference: the formulas and constants are identical to the NumPy edition. Separately, the analytic model agrees with the *production* reference to a few percent — the honest gap between an analytic fit and the detailed cross-section tables, the same gap the NumPy edition's exact KAPP engine closes to the bit (and which the `kgpu` continuum module ports for production use).

**Where this goes next.** With a continuous opacity on the GPU — built on Lecture 2's per-ion populations, fully vectorized over depth *and* wavelength, and validated both against the production reference and against its own fp64 twin — the next lecture carves the **spectral lines** into this continuum floor. Lines add a third axis (the line list) to the batch, and the same broadcasting discipline scales to it: depth $\times$ wavelength $\times$ line, all on the GPU.""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT.relative_to(BOOK)} ({len(cells)} cells)")
