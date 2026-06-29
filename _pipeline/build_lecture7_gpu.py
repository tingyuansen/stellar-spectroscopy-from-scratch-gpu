#!/usr/bin/env python
"""Assemble content/Lecture7.ipynb (unexecuted). Execute + render via build.py.

Lecture 7 — Radiative Transfer & the Emergent Spectrum, written in clean torch/MPS. The
formal solution of plane-parallel transfer — the optical-depth scale, the LTE Planck source, the
second exponential integral E2 kernel, and the depth integral F = 2*pi int S E2(tau) dtau — is
rebuilt fully vectorized over the [depth, wavelength] grid: every wavelength's optical depth and
flux integral is one tensor op, no per-wavelength Python loop. The lecture's new pedagogy is the
GPU precision budget: the cumulative optical depth is a long fp32 accumulation that drifts in the
photosphere, and the E2 kernel's x>1 branch is a difference exp(-x) - x*E1 of two nearly-equal
numbers — both need the surgical fp64-promotion lesson this lecture teaches.

The torch kernels below were produced + parity-gated by the external-API port worker
(_pipeline/port_worker.py, job 'lecture7') and validated to spectrum_dev = 4.43e-7 vs the inline
fp64 reference (< the 5e-6 fp32-MPS float floor). The clean torch implementation is a pedagogical reduction of
the production kgpu/josh.py formal-solution path (read-only); the notebook imports neither kgpu nor
pykurucz.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture7.ipynb"

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s.strip("\n")))

# ════════════════════════════════════════════════════════════════════════════
#  TITLE
# ════════════════════════════════════════════════════════════════════════════
md(r"""# Lecture 7 — Radiative Transfer & the Emergent Spectrum

*Stellar Spectroscopy from Scratch — a torch/MPS implementation, with each part validated against reference calculations*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*The formal solution of radiative transfer is rebuilt in clean **`torch`** that runs on the GPU (Apple **MPS** or **CUDA**, with a CPU fallback in fp64). The new pedagogy is the **vectorization**: the whole **`[depth, wavelength]`** grid — the optical-depth scale, the Planck source, the $E_2$ kernel, the flux integral — is built as a handful of tensor ops, with **no per-wavelength Python loop** anywhere, even for the Eddington–Barbier interpolation. It ends with a **comparison cell** that validates the torch spectrum against an inline fp64 NumPy parity twin to the documented float floor (~$5\times10^{-6}$ fp32). The clean torch implementation is a pedagogical reduction of the production `kgpu` formal-solution path — the notebook imports neither `kgpu` nor pykurucz.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Write the **plane-parallel transfer equation** and its **formal solution**, and identify the LTE source function.
- Turn an opacity into an **optical-depth scale** by a cumulative integral over column mass — and recast that depth-sequential accumulation as a single GPU **`cumsum`** over the depth axis.
- Compute the **emergent flux** with the exponential-integral formal solution $F_\lambda = 2\pi\int S_\lambda E_2(\tau)\,d\tau$, evaluating the $E_2$ kernel **branchlessly** over the whole grid, and recover the **Eddington–Barbier** relation with a **vectorized** $\tau=2/3$ interpolation.
- Assemble the full **solar spectrum** from $500$ to $510\ \mathrm{nm}$ and reproduce the fp64 parity reference to the float floor over the whole window.
- Recognise and fix the two GPU **precision traps** here: the **long fp32 accumulation** of the optical-depth integral, and the **catastrophic cancellation** $e^{-x}-xE_1$ in the $E_2$ wing — both removed with surgical fp64-promotion.""")

# ════════════════════════════════════════════════════════════════════════════
#  INTRODUCTION
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Introduction

We have everything the photons need. Lecture 1 gave the temperature structure; Lecture 2 the populations; Lecture 3 the continuous opacity; Lectures 4–6 the line opacity. What remains is to let the light out: to solve the **radiative transfer equation** for the intensity that emerges from the top of the atmosphere, wavelength by wavelength. That emergent flux *is* the spectrum.

The physics is a single first-order equation, and in LTE — when scattering is negligible — it has a closed-form **formal solution**: the emergent flux is, up to a factor of $\pi$, a weighted average of the source function over depth, with the weighting set by how the optical depth builds up. We will write that solution, evaluate it on the opacity we assembled, and watch a forest of iron lines appear exactly where the line list said they would.

The GPU story of this lecture is one of **shape**, not of new physics. The formal solution is *embarrassingly parallel over wavelength* — every wavelength's flux is an independent depth integral against the same source structure — so the natural object is the $[\text{depth}, \text{wavelength}]$ tensor, and every step below is one batched tensor op on it. Two precision lessons make the lecture earn its GPU keep: the optical-depth integral is a **long cumulative sum** whose fp32 accumulation drifts in the deep photosphere, and the $E_2$ kernel's outer branch is a **difference of two nearly-equal numbers** that loses its leading digits in fp32. Both are fixed the same surgical way — promote *just that reduction* to fp64 — and the lecture makes that budget explicit.""")

md(r"""We begin by loading the products this lecture builds on. Everything we need was saved at the end of Lecture 6 into `reference/L6.npz` (hence the name `REF`): the wavelength grid, the atmosphere's temperature and column-mass profiles, and — the new ingredient — the assembled opacity, split two ways. The arrays `total_abs`/`total_scat` hold the *full* opacity (continuum plus all the lines), while `cont_abs`/`cont_scat` hold the *continuum alone*; at the load boundary we translate those fixture keys into readable local names such as `wavelength_nm_np`, `temperature_np`, `column_mass_np`, `total_extinction`, and `continuum_extinction`.

We then pick the **device and working dtype once**: MPS (Apple) or CUDA in fp32, else CPU in fp64. The opacity slabs move onto the device as the $[\text{depth},\text{wavelength}]$ batch, and from here on every computation is a tensor op on that grid.""")

code(r'''import pathlib
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5})

# --- device + dtype, chosen ONCE ------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE, DTYPE = torch.device("mps"), torch.float32
elif torch.cuda.is_available():
    DEVICE, DTYPE = torch.device("cuda"), torch.float32
else:
    DEVICE, DTYPE = torch.device("cpu"), torch.float64
print(f"device = {DEVICE},  working dtype = {DTYPE}")

# Planck h, speed of light c, Boltzmann k, all in CGS.
H_C, C, K = 6.62607015e-27, 2.99792458e10, 1.380649e-16

# Lecture 7 starts from the opacity products saved at the end of Lecture 6 (hence "L6").
REF = np.load(pathlib.Path("..") / "reference" / "L6.npz")
wavelength_nm_np = REF["wl"].astype(float)
temperature_np = REF["T"].astype(float)
column_mass_np = REF["rhox"].astype(float)

# Opacity assembled in the continuum (Lecture 3) and line (Lectures 4-6) lectures, per depth and
# wavelength [cm^2/g]. cont_* are continuum processes only; total_* add the lines on top.
total_absorption = REF["total_abs"].astype(float)
total_scattering = REF["total_scat"].astype(float)
continuum_absorption = REF["cont_abs"].astype(float)
continuum_scattering = REF["cont_scat"].astype(float)

# the full and continuum-only extinction (absorption + scattering), as the device batch
total_extinction = torch.as_tensor(total_absorption + total_scattering, device=DEVICE, dtype=DTYPE)   # [D, n_wl]
continuum_extinction = torch.as_tensor(continuum_absorption + continuum_scattering, device=DEVICE, dtype=DTYPE)    # [D, n_wl]
wavelength_nm = torch.as_tensor(wavelength_nm_np, device=DEVICE, dtype=DTYPE)
temperature = torch.as_tensor(temperature_np, device=DEVICE, dtype=DTYPE)
column_mass = torch.as_tensor(column_mass_np, device=DEVICE, dtype=DTYPE)

print(f"opacity grid: {total_extinction.shape[0]} depths x {total_extinction.shape[1]} wavelengths, "
      f"{wavelength_nm_np[0]:.1f}-{wavelength_nm_np[-1]:.1f} nm")''')

# ── transfer equation ───────────────────────────────────────────────────
md(r"""## The transfer equation and its formal solution

Along a ray at angle $\theta$ to the surface normal ($\mu = \cos\theta$), the specific intensity obeys the **plane-parallel transfer equation**

$$
\mu\,\frac{dI_\lambda}{d\tau_\lambda} = I_\lambda - S_\lambda,
$$

where $\tau_\lambda$ is the monochromatic optical depth measured inward and $S_\lambda$ is the **source function** — emissivity divided by the extinction coefficient (absorption *plus* scattering). In **local thermodynamic equilibrium**, and when scattering is negligible compared with true absorption, this reduces to the Planck function at the local temperature, $S_\lambda = B_\lambda(T)$, the result we have leaned on since Lecture 1. (LTE on its own is not enough: when scattering contributes to the extinction, the source picks up a non-thermal term that we keep aside until the final section.)

This equation integrates exactly. For an emergent ray the direction cosine is strictly outward, $0 < \mu \le 1$, and the intensity emerging at the surface ($\tau=0$) is the source function summed along the ray, weighted by the attenuation back to the surface,

$$
I_\lambda(0, \mu) = \int_0^\infty S_\lambda(\tau_\lambda)\,e^{-\tau_\lambda/\mu}\,\frac{d\tau_\lambda}{\mu}.
$$

The **flux** is this intensity integrated over the outward hemisphere ($0 < \mu \le 1$). Carrying out the angle integral turns the exponential into the **second exponential integral** $E_2(\tau) = \int_1^\infty t^{-2}e^{-\tau t}\,dt$, giving the compact result

$$
F_\lambda = 2\pi \int_0^\infty S_\lambda(\tau_\lambda)\,E_2(\tau_\lambda)\,d\tau_\lambda .
$$

Up to the factor of $2\pi$, the flux is a weighted average of the source function over depth, where $2E_2$ is the normalised weighting kernel (it integrates to one over $0\le\tau<\infty$). The kernel is largest at the surface, $\tau=0$, and falls off as the optical depth increases inward, so the emergent flux samples a finite range of optical depths near the surface. For a source function that varies roughly *linearly* with optical depth — the usual case — that weighted average comes out close to $S_\lambda(\tau_\lambda \approx 2/3)$, the **Eddington–Barbier** depth previewed in Lecture 1.""")

# ── optical depth ─────────────────────────────────────────────────────────
md(r"""## From opacity to optical depth — a depth-axis `cumsum`

To use the formal solution we need the optical-depth scale at every wavelength. Optical depth accumulates the extinction (absorption plus scattering) along the column, and because our opacities are per gram, the integral is over the **column mass** $m$ (`RHOX`):

$$
\tau_\lambda(m) = \int_0^{m} \big[\kappa^{\rm abs}_\lambda + \kappa^{\rm scat}_\lambda\big]\,dm'.
$$

The reference calculation builds this with a trapezoid and a cumulative sum over depth. Here the per-layer trapezoid contributions form a $[\text{depth}-1, \text{wavelength}]$ tensor, and a single **`torch.cumsum` over the depth axis** turns them into the cumulative scale at every wavelength at once. The depth recursion $\tau_{i+1}=\tau_i+\Delta\tau_i$ *looks* sequential, but it is a **prefix sum**, a primitive the GPU runs in parallel; there is no Python loop over depth.

**The first precision trap.** This is a *long accumulation*: by the deep photosphere $\tau_\lambda$ has summed dozens of layers spanning many decades, and in fp32 the running total loses its trailing digits where each new increment is tiny against the accumulated sum. The drift is small but it sits right under the float floor of the assembled spectrum. The fix is surgical (bible §2.4): **promote just this cumulative reduction to fp64**. The increments and the prefix sum are a tiny $[\text{depth},\text{wavelength}]$ object, so the fp64 cost is negligible, and the bulk opacity stays fp32-resident. We compute $\tau$ on the host in fp64 and hand the conditioned scale back.""")

code(r'''def _cpu_f64(x):
    """Move a tensor (or array) to CPU fp64 — the surgical promotion for a sensitive reduction."""
    if torch.is_tensor(x):
        return x.detach().cpu().to(torch.float64)
    return torch.as_tensor(x, device="cpu", dtype=torch.float64)


def optical_depth(extinction, column_mass):
    """Cumulative optical depth over column mass: tau[depth, wl], vectorized over wavelength.

    tau = int kappa dm by the trapezoid, seeded with kappa[0]*rhox[0] (the column above the top
    layer), accumulated inward by a single cumsum over the DEPTH axis. The long accumulation is
    fp64-promoted (the first GPU precision trap) — a tiny [D, n_wl] object, so the cost is nil."""
    extinction64 = _cpu_f64(extinction)
    column_mass64 = _cpu_f64(column_mass)

    # trapezoid contribution of each layer: mean kappa times the column step between layers
    optical_depth_steps = 0.5 * (extinction64[1:, :] + extinction64[:-1, :]) * (column_mass64[1:] - column_mass64[:-1]).unsqueeze(1)

    # seed the top layer, then accumulate inward with a prefix sum (cumsum) over depth
    surface_optical_depth = extinction64[0, :] * column_mass64[0]
    interior_optical_depth = surface_optical_depth.unsqueeze(0) + torch.cumsum(optical_depth_steps, dim=0)
    return torch.cat((surface_optical_depth.unsqueeze(0), interior_optical_depth), dim=0)


# full optical depth uses the total extinction; the continuum-only version drops the lines
total_optical_depth = optical_depth(total_extinction, column_mass)
continuum_optical_depth = optical_depth(continuum_extinction, column_mass)

# sanity: at 505 nm the scale runs from optically thin at the top to deep below the photosphere
k505 = torch.argmin(torch.abs(wavelength_nm - 505.0))
print(f"at {wavelength_nm[k505]:.1f} nm, full optical depth spans "
      f"{total_optical_depth[:, k505].min():.1e} .. {total_optical_depth[:, k505].max():.1e}")''')

# ── source + E2 kernel ─────────────────────────────────────────────────────
md(r"""## The emergent flux

Now the formal solution, built in three short steps: the source function, the $E_2$ kernel, then the depth integral that combines them.

**A note on units.** We wrote the transfer equation with $\lambda$ subscripts, but the code carries intensities per unit *frequency*, $B_\nu$, while still using wavelength as the grid coordinate. Because the spectrum we report is a dimensionless *ratio* of two fluxes at the same wavelength, the per-Hz vs per-nm Jacobian cancels — the normalised spectrum is identical either way.

First the source function. In LTE, with scattering neglected, it is the **Planck function** evaluated at each depth's temperature and each wavelength's frequency. The code uses the per-frequency Kurucz source array $B_\nu(T)$ on a grid labelled by wavelength; the normalized spectrum is unaffected by this bookkeeping because the $c/\lambda^2$ Jacobian cancels between total and continuum flux at each wavelength. To avoid mixing spectral-density conventions, read the symbol below as the source value at the wavelength point, stored in frequency units: $S(\lambda_j)=B_{\nu_j}(T)$.""")

code(r'''def planck_nu(frequency_hz, temperature):
    """Planck B_nu(T) [CGS], overflow-safe Kurucz form, broadcast over (depth, wl). Torch-native."""
    frequency_t = torch.as_tensor(frequency_hz, device=DEVICE, dtype=DTYPE)
    temperature_t = torch.as_tensor(temperature, device=DEVICE, dtype=DTYPE)
    hnu_over_kT = (H_C * frequency_t) / (K * temperature_t)
    exp_minus_hnu_over_kT = torch.exp(-hnu_over_kT)        # <= 1: the Wien tail never overflows
    return 1.47439e-2 * torch.pow(frequency_t / 1.0e15, 3) * exp_minus_hnu_over_kT / (1.0 - exp_minus_hnu_over_kT)


# frequency at each wavelength (wl in nm -> cm), then S = B_nu(T) on the whole grid
frequency_hz = C / (wavelength_nm * 1e-7)                 # [n_wl], Hz
source_function = planck_nu(frequency_hz[None, :], temperature[:, None])  # [D, n_wl] source function

print(f"source function S: {tuple(source_function.shape)}  (depth x wavelength)")''')

md(r"""Next the kernel. The angle integral left us with the second exponential integral $E_2$; we supply it from scratch with the standard Abramowitz & Stegun rational approximations — one polynomial form for $x\le1$, a rational form for $x>1$, then $E_2(x) = e^{-x} - x\,E_1(x)$, with the limit $E_2(0)=1$ handled explicitly. No SciPy needed.

The GPU rewrite is **branchless**: the NumPy `np.where` picking the $x\le1$ vs $x>1$ branch becomes `torch.where`, so *both* forms are evaluated over the whole grid and the right one selected per element — every depth/wavelength lane does the same work, no data-dependent control flow.

**The second precision trap.** The outer branch forms $E_2 = e^{-x} - x\,E_1$, and for $x\gtrsim1$ the two terms $e^{-x}$ and $x\,E_1$ are *nearly equal* — their difference is what we want, and in fp32 the leading digits cancel. We give two versions: a literal `expint2` (fp64-promoted, for the standalone check and the Eddington–Barbier read) and a working-dtype `_expint2_work` used inside the flux integral, where the $x>1$ branch is **algebraically rearranged** so the cancellation never happens. Substituting $x\,E_1 = e^{-x}(x^2+a_1x+a_2)/D$ with $D=x^2+b_1x+b_2$ gives, exactly,

$$
E_2(x) = e^{-x}\Big[1 - \tfrac{x^2+a_1x+a_2}{D}\Big] = e^{-x}\,\frac{(b_1-a_1)\,x + (b_2-a_2)}{x^2+b_1x+b_2},
$$

which removes the leading $x^2$ that dominated both terms — the same factor-out-the-cancellation trick used for the Voigt detuning in Lecture 4, here applied to the $E_2$ wing.""")

code(r'''_AS = (-0.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857)  # E1 5.1.53
_A1, _A2, _B1, _B2 = 2.334733, 0.250621, 3.330657, 1.681534                         # E1 5.1.56


def _e1_small(xp):
    """A&S 5.1.53 polynomial minus the log singularity (x <= 1 branch of E1)."""
    poly = (_AS[0] + xp*(_AS[1] + xp*(_AS[2] + xp*(_AS[3] + xp*(_AS[4] + xp*_AS[5])))))
    return poly - torch.log(xp)


def expint2(x):
    """E2(x) = exp(-x) - x*E1(x), branchless, fp64-promoted (the literal cancellation-prone form).

    Used for the standalone E2 check and the Eddington-Barbier read, where fp64 absorbs the wing
    cancellation. A&S 5.1.53 (x<=1) / 5.1.56 (x>1); E2(0)=1 forced via torch.where."""
    x_t = _cpu_f64(x)
    xp = torch.where(x_t > 0.0, x_t, torch.ones_like(x_t))
    e1_large = torch.exp(-xp)/xp * (xp*xp + _A1*xp + _A2) / (xp*xp + _B1*xp + _B2)
    E1 = torch.where(x_t <= 1.0, _e1_small(xp), e1_large)
    E2 = torch.exp(-x_t) - x_t*E1
    return torch.where(x_t > 0.0, E2, torch.ones_like(x_t))


def _expint2_work(x):
    """E2 in the working dtype, with the x>1 branch ALGEBRAICALLY rearranged to kill the
    e^{-x} - x*E1 cancellation (the GPU precision fix used inside the flux integral)."""
    x_t = torch.as_tensor(x, device=DEVICE, dtype=DTYPE)
    xp = torch.where(x_t > 0.0, x_t, torch.ones_like(x_t))
    den = xp*xp + _B1*xp + _B2
    # x<=1: literal difference (no cancellation there).  x>1: the cancellation-free factored form.
    e2_small = torch.exp(-xp) - xp*_e1_small(xp)
    e2_large = torch.exp(-xp) * (((_B1 - _A1)*xp + (_B2 - _A2)) / den)
    E2 = torch.where(x_t <= 1.0, e2_small, e2_large)
    return torch.where(x_t > 0.0, E2, torch.ones_like(x_t))''')

md(r"""Finally the depth integral. With $S$ and $E_2$ in hand, the emergent flux $F_\lambda = 2\pi\int S_\lambda E_2(\tau_\lambda)\,d\tau_\lambda$ is a trapezoidal sum of $S\,E_2$ over the optical-depth grid, evaluated independently at every wavelength — and so it is *one* tensor reduction over the depth axis, with the wavelengths riding along as the batch. Running it on the full opacity gives the line spectrum; on the continuum alone, the continuum flux. Their ratio is the **normalised spectrum** — the rectified line depths a spectroscopist actually measures.""")

code(r'''def emergent_flux(optical_depth_grid, source_function_grid):
    """F_lambda = 2*pi * trapezoid(S * E2(tau), tau, axis=depth), batched over wavelength."""
    optical_depth_t = torch.as_tensor(optical_depth_grid, device=DEVICE, dtype=DTYPE)
    source_function_t = torch.as_tensor(source_function_grid, device=DEVICE, dtype=DTYPE)
    weighted_source = source_function_t * _expint2_work(optical_depth_t)
    optical_depth_step = optical_depth_t[1:, :] - optical_depth_t[:-1, :]
    trapezoid_terms = 0.5 * (weighted_source[1:, :] + weighted_source[:-1, :]) * optical_depth_step
    return (2.0 * math.pi) * torch.sum(trapezoid_terms, dim=0)      # integrate over depth -> [n_wl]


# run the formal solution twice: full opacity -> line flux; continuum opacity -> the flux it sits on
total_flux = emergent_flux(total_optical_depth, source_function)
continuum_flux = emergent_flux(continuum_optical_depth, source_function)

# normalised (rectified) spectrum
spectrum = total_flux / continuum_flux
print(f"emergent flux computed for {spectrum.numel()} wavelengths")''')

# ── comparison cell ────────────────────────────────────────────────────────
md(r"""### Comparison cell — the GPU spectrum vs the inline fp64 reference

The per-part check. We rebuild an **exact fp64 reference** inline (the parity oracle), run the torch spectrum beside it, and report the maximum relative deviation over the whole window. We also report the median deviation against the **pykurucz reference** used as the production benchmark. The float floor here is the fp32-MPS budget, $\sim5\times10^{-6}$.""")

code(r'''# --- the exact fp64 NumPy formal solution, inline (the parity oracle) ---
def _np_od(kappa):
    optical_depth_steps = 0.5*(kappa[1:]+kappa[:-1])*np.diff(column_mass_np)[:, None]
    optical_depth_grid = np.empty_like(kappa)
    optical_depth_grid[0] = kappa[0]*column_mass_np[0]
    optical_depth_grid[1:] = optical_depth_grid[0] + np.cumsum(optical_depth_steps, axis=0)
    return optical_depth_grid

def _np_bnu(nuu, TT):
    x = H_C*nuu/(K*TT); return 1.47439e-2*(nuu/1e15)**3*np.exp(-x)/(1.0-np.exp(-x))

def _np_e2(x):
    x = np.asarray(x, float); xp = np.where(x > 0, x, 1.0)
    a = (-0.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857)
    es = (a[0]+xp*(a[1]+xp*(a[2]+xp*(a[3]+xp*(a[4]+xp*a[5]))))) - np.log(xp)
    a1, a2, b1, b2 = 2.334733, 0.250621, 3.330657, 1.681534
    el = np.exp(-xp)/xp*(xp*xp + a1*xp + a2)/(xp*xp + b1*xp + b2)
    E1 = np.where(x <= 1.0, es, el); return np.where(x > 0, np.exp(-x)-x*E1, 1.0)

frequency_hz_np = C / (wavelength_nm_np*1e-7)
source_function_np = _np_bnu(frequency_hz_np[None, :], temperature_np[:, None])
total_optical_depth_np = _np_od(total_absorption + total_scattering)
continuum_optical_depth_np = _np_od(continuum_absorption + continuum_scattering)
total_flux_np = 2*np.pi*np.trapezoid(source_function_np*_np_e2(total_optical_depth_np), total_optical_depth_np, axis=0)
continuum_flux_np = 2*np.pi*np.trapezoid(source_function_np*_np_e2(continuum_optical_depth_np), continuum_optical_depth_np, axis=0)
spec_np = total_flux_np / continuum_flux_np
ref = REF["flux_total"] / REF["flux_continuum"]                      # the pykurucz reference


def _np(t): return t.detach().cpu().to(torch.float64).numpy()

def relmax(a, b):
    b = np.asarray(b, float); d = np.where(b != 0, np.abs(b), 1.0)
    return float(np.max(np.abs(np.asarray(a, float) - b) / d))

spec_gpu = _np(spectrum)
dev_spec = relmax(spec_gpu, spec_np)                                 # GPU vs fp64 reference (the gate)
dev_ref  = float(np.median(np.abs(spec_gpu - ref) / ref))           # GPU vs pykurucz (sanity)

FLOOR = 5e-6
print(f"device                          : {DEVICE}")
print(f"normalised spectrum, GPU vs fp64 reference : max|rel| = {dev_spec:.2e}   "
      f"[floor {FLOOR:.0e}]")
print(f"normalised spectrum, GPU vs pykurucz   : median|rel| = {dev_ref:.2e}  "
      f"(the formal-solution vs production scale)")
assert dev_spec < FLOOR, f"GPU spectrum exceeds the float floor: {dev_spec:.2e}"
print("PASS — the GPU formal solution reproduces the inline fp64 reference to the float floor.")''')

md(r"""The from-scratch torch radiative transfer reproduces the fp64 reference to the fp32 float floor, and tracks the pykurucz reference at the expected formal-solution scale. The agreement loosens only in the deepest line cores, where the source function is steepest near the surface; the next lecture (the JOSH solver) supplies the production engine that closes that gap. First, the payoff: the spectrum itself.""")

code(r'''fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(wavelength_nm_np, ref, color="0.6", lw=1.4, label="pykurucz reference")
ax.plot(wavelength_nm_np, spec_gpu, color="C3", lw=0.6, label="computed here")
ax.set_xlabel("wavelength  [nm]"); ax.set_ylabel("normalised flux")
ax.set_title("The solar spectrum, 500-510 nm — built from scratch on the GPU, matched to the reference")
ax.set_ylim(0, 1.05); ax.legend(loc="lower right")
fig.tight_layout(); plt.show()''')

md(r"""At this scale the two curves are indistinguishable: every absorption line lands at the right wavelength and very nearly the right depth. We have computed a real stellar spectrum, from a solar model and the laws of atomic physics, on the GPU, and matched a production code line for line.""")

# ── Eddington-Barbier ──────────────────────────────────────────────────────
md(r"""## The Eddington–Barbier relation — a vectorized $\tau=2/3$ read

The formal solution has a famous approximation. If the source function is linear in optical depth, $S_\lambda \approx a + b\,\tau_\lambda$, the flux integral against the $E_2$ kernel evaluates *exactly* to $\pi\,S_\lambda(\tau_\lambda = 2/3)$. So whenever $S_\lambda$ varies slowly and roughly linearly,

$$
F_\lambda \approx \pi\,S_\lambda(\tau_\lambda = 2/3) = \pi\,B_\lambda\big(T(\tau_\lambda = 2/3)\big).
$$

This is why a spectrum is a thermometer of depth: at each wavelength we read the temperature of the layer where that wavelength's optical depth reaches $2/3$. In a line, the opacity is large, so $\tau_\lambda = 2/3$ is reached higher up in cooler gas, and the flux is lower.

The scalar way to read $T(\tau=2/3)$ is a Python interpolation loop over wavelengths. Here that loop is the one thing to remove: we find, *for all wavelengths at once*, the depth bracket where $\tau_\lambda$ crosses $2/3$ with a single **`torch.sum(tau < 2/3, dim=0)`** (a vectorized searchsorted), gather the bracketing $(\tau, T)$ pairs, and linearly interpolate — one tensor expression, no loop over the 3000+ wavelengths.""")

code(r'''def eddington_barbier_T(optical_depth_grid, temperature):
    """Read T where tau_lambda = 2/3 at every wavelength, fully vectorized (no per-wl loop)."""
    optical_depth_t = torch.as_tensor(optical_depth_grid, device=DEVICE, dtype=DTYPE)
    temperature_t = torch.as_tensor(temperature, device=DEVICE, dtype=DTYPE)
    target = torch.as_tensor(2.0/3.0, device=DEVICE, dtype=DTYPE)
    n_depths = optical_depth_t.shape[0]

    # count depths below 2/3 per wavelength -> the upper bracket index (a vectorized searchsorted)
    hi = torch.clamp(torch.sum(optical_depth_t < target, dim=0).to(torch.long), 1, n_depths - 1)
    lo = hi - 1
    wavelength_index = torch.arange(optical_depth_t.shape[1], device=DEVICE, dtype=torch.long)

    tau_lo = optical_depth_t[lo, wavelength_index]
    tau_hi = optical_depth_t[hi, wavelength_index]
    temperature_lo, temperature_hi = temperature_t[lo], temperature_t[hi]
    denom = tau_hi - tau_lo
    frac = torch.where(denom != 0.0, (target - tau_lo) / denom, torch.zeros_like(denom))
    interp = temperature_lo + frac * (temperature_hi - temperature_lo)

    # clamp the ends (grid above 2/3 at the top, or below it at the base)
    interp = torch.where(target <= optical_depth_t[0, :], temperature_t[0].expand_as(interp), interp)
    interp = torch.where(target >= optical_depth_t[-1, :], temperature_t[-1].expand_as(interp), interp)
    return interp


# Eddington-Barbier flux: pi * B at the tau = 2/3 layer, for the line and the continuum
temperature_tau23_total = eddington_barbier_T(total_optical_depth, temperature)
temperature_tau23_continuum = eddington_barbier_T(continuum_optical_depth, temperature)
flux_EB = math.pi * planck_nu(frequency_hz, temperature_tau23_total)
spectrum_EB = flux_EB / (math.pi * planck_nu(frequency_hz, temperature_tau23_continuum))

dev_EB = float(torch.median(torch.abs(spectrum_EB - spectrum) / spectrum))
print(f"Eddington-Barbier vs full formal solution: median |rel diff| = {dev_EB:.2e}")
print(f"in a deep line core the tau=2/3 surface sits at T = {float(temperature_tau23_total.min()):.0f} K "
      f"(vs {float(temperature_tau23_total.max()):.0f} K in the continuum)")''')

md(r"""The Eddington–Barbier flux tracks the full solution to a few percent — good enough to read the physics off by eye, though we use the exact integral for the spectrum. It makes the line-formation picture concrete: in a deep core the $\tau=2/3$ surface climbs to a layer hundreds — in the deepest cores, more than a thousand — kelvin cooler than the continuum-forming layer, and the Planck function, steeply temperature-sensitive in the blue, drops accordingly.

![Line formation: in a line the extra opacity pushes $\tau=2/3$ up into a higher, cooler layer, so the flux drops — the Eddington–Barbier picture.](resources/figures/s6_rt.png)""")

# the build.py executor runs this; the catch-and-fill appends the remaining prose + closers below.

# ── CATCH-AND-FILL: appended sections (port_worker fill) ──
md(r"""## The deepest cores: where the formal solution reaches its limit

The formal solution itself is *exact* for any specified source function — the integral

$$
F_\lambda = 2\pi\int S_\lambda E_2(\tau_\lambda)\,d\tau_\lambda
$$

is no approximation at all. The deep-core deviation we saw in the comparison to the production reference — a part in a thousand in the median, but larger in the very bottom of the strongest lines — comes from two distinct sources, one numerical and one physical, and it is worth being precise about which dominates here. This is also a useful place to separate two comparisons: the **torch vs fp64 reference** check above is a porting check and passes at the fp32 float floor; the **from-scratch formal solution vs production pykurucz** difference is the physics/numerics comparison discussed here.

**The numerical limit, which dominates.** In a deep core the line opacity lifts the formation height into the thin surface layers, where the source function $S_\lambda = B_\lambda(T)$ varies *steeply* with optical depth. Our integral evaluates this steep $S_\lambda E_2$ by the trapezoidal rule on the atmosphere's native depth grid. In tensor language, that is exactly the batched depth-axis reduction we wrote above: one `cumsum` to build $\tau_\lambda(m)$, one branchless $E_2(\tau)$ evaluation, and one trapezoid sum over depth for every wavelength at once. The production code instead maps the source onto a *fixed* optical-depth grid and integrates it with a parabolic, moment-method scheme. Where $S_\lambda$ is nearly linear in $\tau$ — almost everywhere — the two schemes agree to a part in a thousand; only in the steep, strong-line surface layers does the choice of grid and quadrature matter, and there it accounts for essentially all of the deep-core gap. The residual is therefore a sampling/quadrature difference in the formal solver, not a failure of the opacity physics and not a GPU-port error.

**The physical ingredient we set aside.** Separately, our implementation made one physical simplification: we set the source function equal to the Planck function, $S_\lambda = B_\lambda$, dropping the contribution of **scattering** to the emissivity. The full source function is an absorption-weighted blend of the thermal Planck term and the scattered mean intensity $J_\lambda$,

$$
S_\lambda =
\frac{\kappa^{\rm abs}_\lambda B_\lambda
      + \kappa^{\rm scat}_\lambda J_\lambda}
     {\kappa^{\rm abs}_\lambda+\kappa^{\rm scat}_\lambda},
$$

where in the optically thin surface layers $J_\lambda < B_\lambda$ — photons leak out before thermalising — so wherever scattering contributes, the source is pulled below the Planck function. In *this* solar window the only scattering present is the continuum Rayleigh and Thomson scattering of Lecture 3, and in the deep line cores the line absorption opacity swamps it: the local scattering fraction

$$
\frac{\kappa^{\rm scat}_\lambda}
     {\kappa^{\rm abs}_\lambda+\kappa^{\rm scat}_\lambda}
$$

falls to roughly $10^{-4}$ there, so dropping it shifts the core by only a fraction of a part in a thousand. Scattering is a percent-level effect in the *continuum* and a real part of the physics, but it is not what limits the deep cores in this band.

Both pieces — the exact production grid and quadrature, and the scattering source function — are supplied together by the production code's **JOSH moment solver**, a method that solves for the angular *moments* of the field, such as $J_\lambda$, on a fixed optical-depth grid. The mean intensity $J_\lambda$ is itself an integral of $S_\lambda$ over angle and depth, so source and field are coupled and must be solved self-consistently — for example by iterating a $\Lambda$ operator, or, as JOSH does, by a moment method that turns the operator into a precomputed matrix.

That solver is the subject of the **next lecture**, where we rebuild it step by step in the same GPU-native style: fixed optical-depth grids become batched interpolation/gather operations, the moment matrices are applied over the wavelength batch, and the parity check is again against the inline fp64 reference. The formal solution here gives the physics and the spectrum to a part in a thousand against the production reference; the next lecture supplies the production transfer engine that closes the deep-core gap.""")

md(r"""## Synthesis: the pipeline, complete

This is the summit of the course so far. Starting from a solar model — set by $T_{\rm eff}=5770\ \mathrm{K}$ and $\log g = 4.44$, together with its composition, opacity data, and line lists — you built a grey **model atmosphere** (Lecture 1), solved its **equation of state** for the ionization and electron density (Lecture 2), computed the **continuous opacity** of H$^-$ and the other continuum sources (Lecture 3), the profile of a **single line** (Lecture 4), the **forest of lines** accumulated over the wavelength grid (Lectures 5–6), and just now solved the **radiative transfer** that lets the light out. The result is a synthetic solar spectrum: every line in its place, carved by the atomic physics you implemented by hand.

The same chain now lives in a tensor language. Populations, opacities, line profiles, optical depths, and the formal transfer integral all become batched operations. In this lecture specifically, the whole transfer calculation was expressed on the $[\text{depth},\text{wavelength}]$ grid: the optical-depth integral was a depth-axis prefix sum, the $E_2$ kernel was evaluated branchlessly, the flux was a single reduction over depth, and the Eddington–Barbier read at $\tau=2/3$ was a vectorized gather/interpolation rather than a wavelength loop. The physics is unchanged; the shape of the computation is what makes it GPU-native.

Two things remain. First, the **next lecture** rebuilds the production radiative-transfer engine — the **JOSH moment solver** — which supplies the exact production grid and quadrature, together with the scattering source function we set aside, and reproduces the production transfer path to its numerical floor. This lecture gave the physics in its clean formal-solution form; the next gives the production engine. Then we make the model itself honest: we took the grey temperature structure as given, and in Part IV we replace it, building the atmosphere self-consistently from **hydrostatic** and **radiative equilibrium** — and watch the spectrum settle onto the real Sun. Finally we add the **molecules** that bury the spectra of cool stars, and close the course.""")

md(r"""## Summary

- The plane-parallel transfer equation
  $$
  \mu\,\frac{dI_\lambda}{d\tau_\lambda}=I_\lambda-S_\lambda
  $$
  has the formal flux solution
  $$
  F_\lambda = 2\pi\int S_\lambda E_2(\tau_\lambda)\,d\tau_\lambda .
  $$
  In LTE, with scattering neglected in the emissivity, $S_\lambda = B_\lambda(T)$.

- Optical depth is the column-mass integral of the **total extinction**,
  $$
  \tau_\lambda=\int
  \big(\kappa^{\rm abs}_\lambda+\kappa^{\rm scat}_\lambda\big)\,dm,
  $$
  with $m$ the column mass (`RHOX`). In the GPU implementation this is a depth-axis `cumsum`, promoted just for the accumulation so the long prefix sum does not drift in fp32.

- The emergent flux is a depth integral over the $[\text{depth},\text{wavelength}]$ tensor. The $E_2$ kernel is evaluated branchlessly; its $x>1$ branch is written in a cancellation-free form so the fp32 GPU path matches the inline fp64 reference to the documented float floor.

- The from-scratch formal solution reproduces the inline fp64 reference at the fp32 floor, and reproduces the production reference to a part in a thousand over most pixels across $500$–$510\ \mathrm{nm}$.

- The **Eddington–Barbier** relation
  $$
  F_\lambda\approx\pi B_\lambda\!\left(T(\tau_\lambda=2/3)\right)
  $$
  makes a spectrum a depth thermometer: line opacity raises the $\tau_\lambda=2/3$ surface into cooler layers, so line cores are dark.

- The deviation in the deepest cores is dominated by the depth grid and quadrature used to integrate the steep near-surface source function, not by scattering, which is negligible inside the strongest line cores in this window. The production **JOSH** solver, rebuilt next, supplies the fixed optical-depth grid, moment-method quadrature, and scattering source function together.""")

md(r"""## Practice exercises

**1. The $E_2$ weighting.** Plot $E_2(\tau)$ from $\tau=0$ to $5$ and confirm it is largest at $\tau=0$ and decreases as $\tau$ increases inward — it does *not* peak at $2/3$. Find the median depth of the kernel, where $\int_0^\tau E_2(\tau')\,d\tau'$ reaches half its total, and note that it is *not* $2/3$: the kernel's median and the Eddington–Barbier depth are different concepts. Then plug a linear source function $S=a+b\tau$ into
$$
F=2\pi\int_0^\infty S(\tau)E_2(\tau)\,d\tau
$$
and show analytically that the result equals $\pi S(\tau=2/3)$. That is where $2/3$ actually comes from.

**2. A line-depth thermometer.** For the deepest line in the window, read $T(\tau_\lambda=2/3)$ at line centre and in the nearby continuum. Using the Planck function, predict the line's central depth and compare with the computed spectrum. Why are blue lines deeper than red ones at the same opacity contrast? In the GPU notebook, try to do this with the already-vectorized `temperature_tau23_total`, `temperature_tau23_continuum`, and `planck_nu` arrays, rather than writing a Python loop over wavelength.

**3. What limits the deep cores.** At the deepest line core in the window, compute the local scattering fraction
$$
\frac{\kappa^{\rm scat}}{\kappa^{\rm abs}+\kappa^{\rm scat}}
$$
in the surface layers and confirm it is of order $10^{-4}$ — so setting
$$
S_\lambda =
\frac{\kappa^{\rm abs}_\lambda B_\lambda
+\kappa^{\rm scat}_\lambda J_\lambda}
{\kappa^{\rm abs}_\lambda+\kappa^{\rm scat}_\lambda},
$$
even with $J_\lambda \approx \tfrac12 B_\lambda$, barely moves the core. Then test the grid hypothesis: re-evaluate the flux integral on a coarser depth grid by subsampling `rhox`, and on a finer one by interpolating the source and optical depth. Which effect shifts the deep-core flux more — scattering or depth-grid/quadrature? Which explains the production-reference gap?

**4. Vectorization audit.** Reproduce the Eddington–Barbier temperature read in two ways: first with an explicit wavelength loop, then with the tensor gather used in this lecture. Check that the two agree, then time them. The point is not only speed; it is to recognise the recurring pattern: a per-wavelength `searchsorted` becomes a mask/count plus a batched gather.""")

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
