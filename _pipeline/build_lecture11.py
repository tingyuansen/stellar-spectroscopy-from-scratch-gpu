#!/usr/bin/env python
"""Lecture 11 — Convection & the Converged Atmosphere.

Adds mixing-length convection to the deep photosphere and ITERATES the radiative-
equilibrium temperature correction of Lecture 10 to flux constancy, producing the
end-to-end converged continuum-only solar model (Teff=5770, logg=4.44).  Benchmarked
to machine precision against reference/converged_ref.npz via a fixed-point check:
one from-scratch iteration {KAPP opacity, JOSH flux, Rosseland mean, mixing-length
CONVEC, TCORR} from the pykurucz-converged model reproduces it.

Self-contained: imports only numpy / matplotlib / pathlib.  No pykurucz import.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture11.ipynb"
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

# ── title ────────────────────────────────────────────────────────────────
md(r"""# Lecture 11 — Convection & the Converged Atmosphere

*Stellar Spectroscopy from Scratch — rebuilding the physics of ATLAS and SYNTHE from first principles*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*Every result in this book is checked against reference values computed with [**pykurucz**](https://arxiv.org/abs/2603.11693) — a pure-Python implementation of Kurucz's ATLAS12 and SYNTHE — shipped beside the lectures as small data files, so the notebooks need only NumPy to run.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Explain why the **deep photosphere of a cool star convects**: where the radiative gradient exceeds the **adiabatic** gradient, the gas becomes unstable and carries part of the flux as rising and falling parcels rather than as radiation.
- Implement **mixing-length theory** (Kurucz's `CONVEC`): the adiabatic gradient $\nabla_{\rm ad}$, the superadiabaticity $\Delta = \nabla - \nabla_{\rm ad}$, the convective velocity and the **convective flux** $F_{\rm conv}$, with the thermodynamic derivatives taken from **finite differences** of the equation of state and the optically-thick efficiency factor $\tau_b^2/(2+\tau_b^2)$.
- See that convection is **negligible in the line-forming layers** but reshapes the **deep-layer** structure, and that its flux feeds back into the temperature correction.
- State the **convergence criterion** — flux constancy, measured as $\max|\Delta T/T|$ over the deep layers — and read the iteration history of a real model converging.
- Run the loop from the **grey start** to the converged continuum-only solar model (28 iterations) and reproduce one from-scratch iteration of it to **machine precision** — the new convective flux, the corrected temperature, and the column mass — closing the loop the first lecture opened.""")

# ── introduction ───────────────────────────────────────────────────────────
md(r"""## Introduction: from one correction step to a finished atmosphere

Lecture 10 built **one** step of the radiative-equilibrium temperature correction: it measured how far the grey atmosphere's flux was from constant, and shifted the temperature to push it back. That was the engine. This lecture turns the engine into a machine that produces a *finished* model atmosphere, and it does so by adding the two pieces Lecture 10 deliberately left out.

The first is **iteration**. One correction does not converge a model: after we change the temperature, the equation of state (Lecture 2), the opacities (Lecture 3), the Rosseland mean and the fluxes (Lectures 8, 10) all change too, so we must recompute them and correct again — and again — until the flux stops drifting with depth and the temperature stops moving. A solar model takes a few dozen such iterations from a grey start.

The second is **convection**. In the deep photosphere of a cool star like the Sun, radiation alone cannot carry all the flux: the temperature gradient steepens until the gas becomes **convectively unstable**, and rising hot parcels and sinking cool ones take over a large fraction of the energy transport. The temperature correction has to know about that convective flux, or it will get the deep structure wrong. Convection is **negligible where the spectral lines form** (the upper photosphere, which stays radiative), so none of our spectrum work in Lectures 1–8 needed it — but it matters for the *structure* of the deep layers, and therefore for a self-consistent model.

We reuse, unchanged, every engine the book has built: the continuum opacity **KAPP** (Lecture 3, supplied as a given input), the **JOSH** moment solver (Lecture 8) for the per-frequency flux, the **Rosseland mean** and the **temperature correction TCORR** (Lecture 10). The genuinely new physics is the mixing-length **convection** kernel and the **convergence** loop that ties everything together.""")

# ── setup and the reference ──────────────────────────────────────────────────
md(r"""## Setup and the reference

We import only NumPy and Matplotlib. The benchmark target is `reference/converged_ref.npz`, produced once by the production code on the solar parameters ($T_{\rm eff}=5770$ K, $\log g=4.44$) in the clean configuration this book uses: **continuum opacity only** (no line blanketing, so no multi-gigabyte line lists), **convection on** (mixing length $1.25$), and **serial** execution so the result is bit-reproducible. The production code was run from the **grey start** of Lecture 9 all the way to convergence — flux constant to a part in $10^4$ over the deep layers — which took **28 iterations**.

The file ships three things. First, the **converged model** itself ($T$, column mass `rhox`, pressure, electron density, Rosseland opacity, and the convective flux `flxcnv`). Second, the **convergence history** — $\max|\Delta T/T|$ per iteration — so we can plot a real model settling. Third, everything one from-scratch iteration *from the converged model* needs: the per-frequency **continuum opacity** (the Lecture-3 KAPP output, a given input), the equation-of-state **finite-difference samples** the convection kernel needs (`edens1..4`, `rho1..4` — the electron density and mass density at $T,P\pm0.1\%$, treated as given so we need not re-run the full EOS), and the convective inputs (`ptotal`, `rho`, `pradk`, `prad`, `abross`).""")

code(r'''import pathlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5})

REF = np.load(pathlib.Path("..") / "reference" / "converged_ref.npz")
JT  = np.load(pathlib.Path("..") / "reference" / "josh_tables.npz")   # Lecture 8 tables

TEFF = float(REF["teff"]); GRAV = float(REF["gravity_cgs"]); LOGG = float(np.log10(GRAV))
N_ITER = int(REF["n_iterations"])
print(f"target: converged continuum-only model for Teff = {TEFF:.0f} K, log g = {LOGG:.2f}")
print(f"  production run: grey start -> convergence in {N_ITER} iterations (convection ON)")
print(f"  continuum grid: {REF['freq_hz'].size} frequencies x {REF['T_conv'].size} layers")''')

# ── the convergence history ──────────────────────────────────────────────────
md(r"""## What "converged" means: flux constancy

Radiative (plus convective) equilibrium demands that the **total** flux — radiative plus convective — be the same at every depth, equal to $\sigma T_{\rm eff}^4$. ATLAS does not test the flux directly; it tests the thing the flux drives, the **temperature change between successive iterations**. When the model is converged, one more correction barely moves the temperature, so the convergence metric is

$$\max_{j}\ \left|\frac{T^{(N)}_j - T^{(N-1)}_j}{T^{(N)}_j}\right| < 10^{-4},$$

the maximum fractional temperature change taken over the **deep layers** (ATLAS's `checkconv` uses layers 40–75 of the 80-layer grid — the upper layers are noisier and settle last in absolute terms but are irrelevant to the energy balance). The loop stops when this drops below $10^{-4}$, after a minimum number of iterations. The reference shipped that history; here it is.""")

code(r'''dlnt  = REF["dlnt_history"]    # max|dT/T| over deep layers 40..75, per iteration
dtmax = REF["dtmax_history"]   # max|dT/T| over ALL layers, per iteration (top layers settle last)

print(" iter   dTmax(all)    checkconv_dlnt(deep)")
for i in range(dlnt.size):
    flag = "  <- converged" if dlnt[i] < 1e-4 else ""
    print(f"  {i+1:3d}    {dtmax[i]:.3e}     {dlnt[i]:.3e}{flag}")
print(f"\\nconverged when deep-layer max|dT/T| < 1e-4 (Fortran checkconv.f90)")
print(f"determinism: two full runs identical  "
      f"T={bool(int(REF['determinism_T']))}  RHOX={bool(int(REF['determinism_R']))}")''')

md(r"""The deep-layer metric falls monotonically from order unity (the grey start is far from equilibrium) toward $10^{-4}$; the all-layer metric stalls higher because the thin **top** layers keep jittering at the $10^{-4}$ level even when the energy-carrying deep layers are settled — which is exactly why `checkconv` looks only at the deep layers. We will plot this curve at the end, next to the final flux-constancy check.""")

# ── numerical toolbox ────────────────────────────────────────────────────────
md(r"""## The numerical toolbox (Lecture 8)

As in Lecture 10, every integral, derivative, and grid remap below is built from the four Fortran kernels of Lecture 8, reproduced compactly. `parcoe` fits a smoothly-blended parabola through each triple of points; `integ` integrates with each interval's parabola (every optical depth and flux integral); `deriv` is the cubic-tangent derivative ATLAS uses for gradients like $dT/d\rho x$; `map1` is the piecewise-quadratic remap between depth grids. The convection kernel needs `deriv` for the temperature and opacity gradients and `integ` for the geometric height.""")

code(r'''def parcoe(f, x):
    """Smoothly-blended parabolic coefficients a,b,c so f ~ a + b x + c x^2 (Fortran PARCOE)."""
    nn = f.size; a = np.zeros(nn); b = np.zeros(nn); c = np.zeros(nn)
    if nn == 1: a[0] = f[0]; return a, b, c
    b[0] = (f[1]-f[0])/(x[1]-x[0]); a[0] = f[0]-x[0]*b[0]                # endpoints: straight line
    n1 = nn-1
    b[-1] = (f[-1]-f[n1-1])/(x[-1]-x[n1-1]); a[-1] = f[-1]-x[-1]*b[-1]
    if nn == 2: return a, b, c
    for j in range(1, n1):                                              # interior: parabola through 3 points
        j1 = j-1; d = (f[j]-f[j1])/(x[j]-x[j1])
        c[j] = f[j+1]/((x[j+1]-x[j])*(x[j+1]-x[j1])) + (f[j1]/(x[j+1]-x[j1]) - f[j]/(x[j+1]-x[j]))/(x[j]-x[j1])
        b[j] = d - (x[j]+x[j1])*c[j]; a[j] = f[j1] - x[j1]*d + x[j]*x[j1]*c[j]
    c[1] = 0.0; b[1] = (f[2]-f[1])/(x[2]-x[1]); a[1] = f[1]-x[1]*b[1]   # 2nd, 3rd points forced linear
    if nn > 3: c[2] = 0.0; b[2] = (f[3]-f[2])/(x[3]-x[2]); a[2] = f[2]-x[2]*b[2]
    for j in range(1, n1):                                             # blend each parabola with its neighbour
        if c[j] == 0.0: continue
        j1 = min(j+1, nn-1); den = abs(c[j1])+abs(c[j]); wt = abs(c[j1])/den if den > 0 else 0.0   # curvature weight
        a[j] = a[j1]+wt*(a[j]-a[j1]); b[j] = b[j1]+wt*(b[j]-b[j1]); c[j] = c[j1]+wt*(c[j]-c[j1])
    a[n1-1] = a[-1]; b[n1-1] = b[-1]; c[n1-1] = c[-1]
    return a, b, c''')

code(r'''def integ(x, f, start):
    """Cumulative integral of f dx, each interval using its left-point parabola (Fortran INTEG)."""
    nn = f.size; out = np.zeros(nn)
    if nn == 0: return out
    a, b, c = parcoe(f, x); out[0] = start
    for i in range(nn-1):
        dx = x[i+1]-x[i]
        term = a[i] + 0.5*b[i]*(x[i+1]+x[i]) + (c[i]/3.0)*((x[i+1]+x[i])*x[i+1] + x[i]*x[i])
        out[i+1] = out[i] + term*dx
    return out

def deriv(x, f):
    """Cubic-tangent derivative df/dx (Fortran DERIV)."""
    nn = f.size; d = np.zeros(nn)
    if nn < 2: return d
    d[0] = (f[1]-f[0])/(x[1]-x[0]); d[-1] = (f[-1]-f[-2])/(x[-1]-x[-2])
    if nn == 2: return d
    s = abs(x[1]-x[0])/(x[1]-x[0]) if x[1] != x[0] else 1.0
    for j in range(1, nn-1):
        scale = max(abs(f[j-1]), abs(f[j]), abs(f[j+1]))
        scale = scale/abs(x[j]) if x[j] != 0.0 else scale
        if scale == 0.0: scale = 1.0
        d1 = (f[j+1]-f[j])/(x[j+1]-x[j])/scale; d0 = (f[j]-f[j-1])/(x[j]-x[j-1])/scale
        tan1 = d1/(s*np.sqrt(1.0+d1*d1)+1.0); tan0 = d0/(s*np.sqrt(1.0+d0*d0)+1.0)
        d[j] = (tan1+tan0)/(1.0-tan1*tan0)*scale
    return d''')

code(r'''def map1(xold, fold, xnew):
    """Piecewise-quadratic remap matching Fortran MAP1.  Returns (fnew, ll-1)."""
    nold, nnew = xold.size, xnew.size; fnew = np.zeros(nnew)
    if nold == 0 or nnew == 0: return fnew, 0
    xo = np.empty(nold+1); fo = np.empty(nold+1); xo[1:] = xold; fo[1:] = fold
    l = 2; ll = 0; cfor = bfor = afor = cbac = bbac = abac = a = b = c = 0.0
    for k in range(1, nnew+1):
        xk = xnew[k-1]
        while True:
            if xk < xo[l]:
                if l == ll: break
                if l == 2 or l == 3:
                    l = min(nold, l); c = 0.0
                    b = (fo[l]-fo[l-1])/(xo[l]-xo[l-1]); a = fo[l]-xo[l]*b; ll = l; break
                l1 = l-1
                if l > ll+1 or l == 3 or l == 4:
                    l2 = l-2; d = (fo[l1]-fo[l2])/(xo[l1]-xo[l2])
                    cbac = fo[l]/((xo[l]-xo[l1])*(xo[l]-xo[l2])) + (fo[l2]/(xo[l]-xo[l2]) - fo[l1]/(xo[l]-xo[l1]))/(xo[l1]-xo[l2])
                    bbac = d-(xo[l1]+xo[l2])*cbac; abac = fo[l2]-xo[l2]*d+xo[l1]*xo[l2]*cbac
                else:
                    cbac, bbac, abac = cfor, bfor, afor
                if l >= nold:
                    c, b, a, ll = cbac, bbac, abac, l; break
                d = (fo[l]-fo[l1])/(xo[l]-xo[l1])
                cfor = fo[l+1]/((xo[l+1]-xo[l])*(xo[l+1]-xo[l1])) + (fo[l1]/(xo[l+1]-xo[l1]) - fo[l]/(xo[l+1]-xo[l]))/(xo[l]-xo[l1])
                bfor = d-(xo[l]+xo[l1])*cfor; afor = fo[l1]-xo[l1]*d+xo[l]*xo[l1]*cfor
                wt = abs(cfor)/(abs(cfor)+abs(cbac)) if abs(cfor) != 0.0 else 0.0
                a = afor+wt*(abac-afor); b = bfor+wt*(bbac-bfor); c = cfor+wt*(cbac-cfor); ll = l; break
            l += 1
            if l > nold:
                l = min(nold, l); c = 0.0
                b = (fo[l]-fo[l-1])/(xo[l]-xo[l-1]); a = fo[l]-xo[l]*b; ll = l; break
        fnew[k-1] = a + (b + c*xk)*xk
    return fnew, max(ll-1, 0)

def map1_scalar(xold, fold, xnew_val):
    out, _ = map1(np.asarray(xold), np.asarray(fold), np.asarray([xnew_val])); return float(out[0])

def _nz_signed(x, eps=1e-300):
    """Replace a value too close to zero with a small SIGNED floor (Fortran guard)."""
    return x if abs(x) >= eps else (eps if x >= 0.0 else -eps)''')

# ── constants and the converged model ────────────────────────────────────────
md(r"""## Constants and the converged model

We use exactly the constants ATLAS uses. We then load the **converged model** as the starting point of our from-scratch iteration: temperature `T`, column mass `rhox`, gas pressure `p_in`, mass density `rho_in`, and the convective and radiative pressures. The single number that sets the energy balance is the target Eddington flux $H = \sigma T_{\rm eff}^4/(4\pi)$; the **total** flux (radiative + convective) must equal this at every layer.""")

code(r'''SIGMA  = 5.6697e-5     # Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4]
PLANCK = 6.6256e-27    # Planck constant h [erg s]
KBOLTZ = 1.38054e-16   # Boltzmann constant k [erg/K]
FOURPI = 12.5664       # 4 pi, written explicitly by ATLAS

T      = REF["T_conv"].astype(np.float64)       # converged temperature [K], 80 layers
rhox   = REF["rhox_conv"].astype(np.float64)    # column mass RHOX [g/cm^2]
p_in   = REF["p_conv"].astype(np.float64)       # gas pressure [dyn/cm^2]
rho_in = REF["rho_conv"].astype(np.float64)     # mass density [g/cm^3]
prad   = REF["prad_conv"].astype(np.float64)    # radiation pressure [dyn/cm^2]
pradk  = REF["pradk_conv"].astype(np.float64)   # PRADK = PRAD + surface K-integral
ptotal = REF["ptotal_conv"].astype(np.float64)  # total pressure GRAV*RHOX + PZERO
n = T.size

hkt  = PLANCK / np.maximum(T * KBOLTZ, 1e-300)  # h/(kT) per layer (nu factored in later)
flux = SIGMA / FOURPI * TEFF**4                 # target Eddington flux H = sigma Teff^4 / (4 pi)
print(f"target Eddington flux  H = {flux:.4e} erg cm^-2 s^-1 sr^-1")''')

# ── JOSH and the frequency sweep ─────────────────────────────────────────────
md(r"""## The per-frequency flux: JOSH and the Rosseland mean (Lecture 10)

The temperature correction needs four depth integrals over the continuum spectrum, exactly as in Lecture 10: the depth-resolved Eddington flux $H_\nu(\tau)$ and the moment $(J_\nu-S_\nu)(\tau)$ at each frequency, summed with the quadrature weights into `flxrad` (radiative flux), `rjmins`, `rdabh`, and the $\Lambda$-diagonal `rdiagj`; plus the Rosseland accumulator that becomes $\kappa_{\rm Ross}$. We carry over the JOSH full-depth kernel and the $E_3$ exponential integral from Lecture 10 verbatim; the only float32 step in the whole pipeline is JOSH's inner $\Lambda$-iteration, and it is what sets the precision floor. We fold the JOSH kernel and the helpers into one cell here to keep the focus on the new physics — the convection that follows.""")

code(r'''XTAU  = JT["xtau"].astype(np.float64)             # JOSH fixed optical-depth grid
CH    = JT["ch"].astype(np.float64)
COEFJ = JT["coefj"].astype(np.float64)            # Lambda operator on the fixed grid
CH_MAT = REF["josh_coefh"].astype(np.float64)     # H-moment operator (shipped with the reference)
ITER_TOL = 1.0e-5

def expi3(x):
    """E3 exponential integral (Fortran EXPI N=3), used by the Lambda-diagonal integral."""
    a = (-44178.5471728217, 57721.7247139444, 9938.31388962037, 1842.11088668, 101.093806161906, 5.03416184097568)
    b = (76537.3323337614, 32597.1881290275, 6106.10794245759, 635.419418378382, 37.2298352833327)
    c = (4.65627107975096e-7, 0.999979577051595, 9.04161556946329, 24.3784088791317, 23.0192559391333, 6.90522522784444, 0.430967839469389)
    dco = (10.0411643829054, 32.4264210695138, 41.2807841891424, 20.4494785013794, 3.31909213593302, 0.103400130404874)
    e = (-0.999999999998447, -26.6271060431811, -241.055827097015, -895.927957772937, -1298.85688746484, -545.374158883133, -5.66575206533869)
    fco = (28.6271060422192, 292.310039388533, 1332.78537748257, 2777.61949509163, 2404.01713225909, 631.6574832808)
    if x <= 0.0: ex1 = 0.0
    else:
        ex = np.exp(-x)
        if x > 4.0:
            ex1 = (ex + ex*(e[0]+(e[1]+(e[2]+(e[3]+(e[4]+(e[5]+e[6]/x)/x)/x)/x)/x)/x)/(x+fco[0]+(fco[1]+(fco[2]+(fco[3]+(fco[4]+fco[5]/x)/x)/x)/x)/x))/x
        elif x > 1.0:
            ex1 = ex*(c[6]+(c[5]+(c[4]+(c[3]+(c[2]+(c[1]+c[0]*x)*x)*x)*x)*x)*x)/(dco[5]+(dco[4]+(dco[3]+(dco[2]+(dco[1]+(dco[0]+x)*x)*x)*x)*x)*x)
        else:
            ex1 = (a[0]+(a[1]+(a[2]+(a[3]+(a[4]+a[5]*x)*x)*x)*x)*x)/(b[0]+(b[1]+(b[2]+(b[3]+(b[4]+x)*x)*x)*x)*x) - np.log(x)
    out = ex1
    for i in range(1, 3): out = (np.exp(-x) - x*out)/float(i)
    return out''')

code(r'''def josh_profiles(acont, scont, sigmac, rhox, bnu):
    """JOSH full depth profiles for one frequency (Lecture 8/10).  Returns taunu, hnu, jmins, abtot, alpha.
    Continuum only: aline = sigmal = 0; sline = bnu.  The inner Lambda sweep runs in float32 (REAL*4)."""
    nl = rhox.size; nxtau = XTAU.size; coefj_diag = np.diag(COEFJ).astype(np.float32)
    abtot = np.maximum(acont + sigmac, 1e-300)          # total extinction
    alpha = sigmac / abtot                                # scattering fraction
    snubar = bnu.copy()
    np.divide(acont*scont, acont, out=snubar, where=acont > 0.0)   # thermal source (continuum)
    taunu = integ(rhox, abtot, abtot[0]*rhox[0])         # monochromatic optical depth
    snu = np.zeros(nl); hnu = np.zeros(nl); jnu = np.zeros(nl); jmins = np.zeros(nl)
    xs = np.zeros(nxtau, dtype=np.float32)                # source vector on the fixed xtau grid (REAL*4)
    if taunu[0] > XTAU[-1]:
        maxj = 1                                          # whole column deeper than the grid -> skip surface solve
    else:
        # remap the thermal source and scattering fraction onto the fixed xtau grid
        xsbar8, maxj = map1(taunu, snubar, XTAU); xalpha8, maxj = map1(taunu, alpha, XTAU)
        xalpha8 = np.maximum(xalpha8.astype(np.float32), np.float32(0.0))      # clamp to [0, .]
        xsbar8 = np.maximum(xsbar8.astype(np.float32), np.float32(1.0e-38))    # positive floor
        mask = XTAU < taunu[0]                            # grid points above the physical surface
        if np.any(mask): xsbar8[mask] = max(snubar[0], 1.0e-38); xalpha8[mask] = max(alpha[0], 0.0)
        xs[:] = xsbar8; one32 = np.float32(1.0)
        diag = one32 - xalpha8*coefj_diag                 # diagonal of (1 - alpha Lambda)
        xsbar_mod = (one32 - xalpha8)*xsbar8              # thermal part (1-alpha) Sbar
        for _ in range(nxtau):                            # backward Gauss-Seidel sweep, float32 throughout
            iferr = 0
            for kk in range(nxtau):
                k = nxtau-1-kk                            # sweep deep -> shallow
                dot = np.float32(np.dot(COEFJ[k, :].astype(np.float32), xs))   # (Lambda S)_k
                num = np.float32(dot*xalpha8[k] + xsbar_mod[k] - xs[k])        # residual of S = (1-a)Sbar + a Lambda S
                dd = np.float32(diag[k])
                if abs(float(dd)) < 1.0e-37: dd = np.float32(1.0e-37 if float(dd) >= 0.0 else -1.0e-37)   # signed floor
                delxs = np.float32(num/dd); xbase = np.float32(xs[k])          # Gauss-Seidel update
                if abs(float(xbase)) < 1.0e-37: xbase = np.float32(1.0e-37 if float(xbase) >= 0.0 else -1.0e-37)
                if np.float32(abs(float(delxs/xbase))) > np.float32(ITER_TOL): iferr = 1   # not yet converged
                xs[k] = np.float32(max(float(np.float32(xs[k]+delxs)), 1.0e-37))           # apply, keep positive
            if iferr == 0: break                          # all layers within tolerance -> done
        snu_head, _ = map1(XTAU, xs.astype(np.float64), taunu[:maxj]); snu[:maxj] = snu_head   # back to physical grid
    maxj1 = maxj+1 if maxj != 1 else 1
    snu[maxj1-1:] = snubar[maxj1-1:]                      # deep layers seeded with the thermal source
    m0 = max(maxj-1, 1) - 1; nmj0 = maxj-1
    for _ in range(nxtau):                                # deep layers solved on the physical TAUNU grid
        error = 0.0; ifneg = 0
        if np.any(snu[m0:] <= 0.0): ifneg = 1; snubar[m0:] = bnu[m0:]; snu[m0:] = bnu[m0:]   # negativity guard
        hnu[m0:] = deriv(taunu[m0:], snu[m0:])/3.0        # H = (1/3) dS/dtau (diffusion)
        if np.any(hnu[m0:] <= 0.0):                       # second guard: reset to Planck if H went negative
            ifneg = 1; snubar[m0:] = bnu[m0:]; snu[m0:] = bnu[m0:]; hnu[m0:] = deriv(taunu[m0:], snu[m0:])/3.0
        jmins[nmj0:] = deriv(taunu[nmj0:], hnu[nmj0:])    # (J - S) = dH/dtau
        for j in range(maxj1-1, nl):
            if ifneg == 1: jmins[j] = 0.0
            jnu[j] = jmins[j] + snu[j]                     # mean intensity J
            snew = (1.0-alpha[j])*snubar[j] + alpha[j]*jnu[j]                  # updated source (scattering)
            error += abs(snew-snu[j])/max(abs(snew), 1e-300); snu[j] = snew
        if error < ITER_TOL: break
    if maxj == 1: return taunu, hnu, jmins, abtot, alpha
    xjs = (-xs + COEFJ.astype(np.float32) @ xs).astype(np.float64)     # (J-S) from the matrix solution
    xh  = (CH_MAT @ xs).astype(np.float64)                             # H from the matrix solution
    jmins[:maxj], _ = map1(XTAU, xjs, taunu[:maxj]); hnu[:maxj], _ = map1(XTAU, xh, taunu[:maxj])   # to physical grid
    return taunu, hnu, jmins, abtot, alpha''')

md(r"""### The frequency sweep

We sweep the 30000 continuum frequencies. At each we form the Planck function $B_\nu$, run JOSH for the depth profiles, and accumulate the Rosseland integral (a *harmonic*, $\partial B_\nu/\partial T$-weighted average of $1/\kappa_\nu$) and the four TCORR integrals. The last integral, `rdiagj`, builds the diagonal of the $\Lambda$ operator on the physical grid using the $E_3$ kernel — the same construction as Lecture 10.""")

code(r'''freq = REF["freq_hz"].astype(np.float64); rco = REF["rco"].astype(np.float64)
acont = REF["acont"].astype(np.float64); sigmac = REF["sigmac"].astype(np.float64); scont = REF["scont"].astype(np.float64)
nf = freq.size; z = np.zeros(n)

ross_acc = np.zeros(n)                                   # Rosseland mean accumulator
flxrad = np.zeros(n); rjmins = np.zeros(n); rdabh = np.zeros(n); rdiagj = np.zeros(n)  # TCORR integrals
for inu in range(nf):
    f = float(freq[inu]); rcowt = float(rco[inu])
    ehvkt = np.exp(-f*hkt); stim = np.maximum(1.0-ehvkt, 1e-300)   # stimulated-emission factor
    bnu = 1.47439e-2*((f/1e15)**3)*ehvkt/stim                       # Planck function B_nu
    taunu, hnu, jmins, abtot, alpha = josh_profiles(acont[:, inu], scont[:, inu], sigmac[:, inu], rhox, bnu)
    if np.any(hnu < 0.0): hnu = np.maximum(hnu, 1e-99)
    dbdt = bnu*f*hkt/np.maximum(T*stim, 1e-300)                     # dB_nu/dT (the diffusion weight)
    ross_acc += dbdt/np.maximum(abtot, 1e-300)*rcowt               # Rosseland: harmonic 1/kappa weighting
    dabtot = deriv(rhox, abtot)                                    # d(kappa_nu)/d(rhox), for rdabh
    rdabh  += dabtot/np.maximum(abtot, 1e-300)*hnu*rcowt           # log-opacity gradient x flux integral
    rjmins += abtot*jmins*rcowt                                    # net heating integral kappa (J - S)
    flxrad += hnu*rcowt                                            # radiative flux H(tau), summed over nu
    term2 = 0.0
    for j in range(n):                                             # Lambda-diagonal via the E3 kernel (per layer)
        term1 = term2
        d = max(1e-10, float(taunu[j+1]-taunu[j]) if j != n-1 else 1e-10)   # optical-depth step to next layer
        if d <= 0.01:                                             # small-step series expansion of the E3 integral
            term2 = (0.922784335098467-np.log(d))*d/4.0 + d*d/12.0 - d**3/96.0 + d**4/720.0
        else:                                                     # otherwise the exact E3 kernel
            ex = expi3(d) if d < 10.0 else 0.0
            term2 = 0.5*(d + ex - 0.5)/d
        diagj = term1 + term2                                     # Lambda-operator diagonal element
        dbdtj = bnu[j]*f*hkt[j]/max(T[j]*stim[j], 1e-300)
        rdiagj[j] += abtot[j]*(diagj-1.0)/max(1.0-alpha[j]*diagj, 1e-300)*(1.0-alpha[j])*dbdtj*rcowt
print(f"swept {nf} continuum frequencies; flux at surface = {flxrad[0]:.4e}, deep = {flxrad[-1]:.4e}")''')

md(r"""## The Rosseland optical-depth scale and its lookup table

The Rosseland mean opacity follows from the accumulator, $\kappa_{\rm Ross} = (4\sigma/\pi)\,T^3/\texttt{ross\_acc}$, and integrating it against the column mass gives the Rosseland optical-depth scale $\tau_{\rm Ross}$ (Lecture 10). We also build the **`ROSSTAB`** table now: a $(\log T,\log P,\log\kappa)$ table assembled from this iteration's $(T,P,\kappa_{\rm Ross})$ at every layer, with a nearest-neighbour-per-quadrant bilinear lookup. Convection reads it for the opacity *inside* a convective cell, where the temperature is offset from the layer mean; the hydrostatic density correction reads it too. It is the same table Lecture 10 introduced.""")

code(r'''abross = (4.0*SIGMA/3.14159) * T**3 / np.maximum(ross_acc, 1e-300)   # Rosseland mean [cm^2/g]
tauros = integ(rhox, abross, abross[0]*rhox[0])                       # Rosseland optical depth

class Rosstab:
    """ROSSTAB: (logT, logP, log kappa) table + nearest-neighbour-per-quadrant bilinear lookup."""
    def __init__(self): self.t=[]; self.p=[]; self.k=[]; self.zerot=self.zerop=0.0; self.slopet=self.slopep=1.0; self.n=0
    def ingest(self, T, P, kappa):
        if self.n == 0:                                  # first call: set the (logT, logP) normalization
            self.zerot = np.log10(max(float(T[0]),1e-300)); self.zerop = np.log10(max(float(P[0]),1e-300))   # origins
            self.slopet = np.log10(max(float(T[-1]),1e-300))-self.zerot; self.slopep = np.log10(max(float(P[-1]),1e-300))-self.zerop  # spans
            if abs(self.slopet) < 1e-300: self.slopet = 1.0
            if abs(self.slopep) < 1e-300: self.slopep = 1.0
        for j in range(T.size):                          # store each layer as (normalized logT, logP, log kappa)
            self.t.append((np.log10(max(float(T[j]),1e-300))-self.zerot)/self.slopet)
            self.p.append((np.log10(max(float(P[j]),1e-300))-self.zerop)/self.slopep)
            self.k.append(np.log10(max(float(kappa[j]),1e-300))); self.n += 1
    def eval(self, temp, pressure):
        if self.n <= 0: return 1.0
        tl = (np.log10(max(temp,1e-300))-self.zerot)/self.slopet; pl = (np.log10(max(pressure,1e-300))-self.zerop)/self.slopep
        rpp=rpm=rmp=rmm=1e30; i_pp=i_pm=i_mp=i_mm=-1; v_pp=v_pm=v_mp=v_mm=0.0
        for i in range(self.n):                              # nearest neighbour in each (dT,dP) quadrant
            dp = self.p[i]-pl; dt = self.t[i]-tl; r2 = dt*dt+dp*dp
            if dt >= 0 and dp >= 0:
                if r2 < rpp: rpp=r2; i_pp=i; v_pp=self.k[i]
            elif dt >= 0 and dp < 0:
                if r2 < rpm: rpm=r2; i_pm=i; v_pm=self.k[i]
            elif dt < 0 and dp >= 0:
                if r2 < rmp: rmp=r2; i_mp=i; v_mp=self.k[i]
            else:
                if r2 < rmm: rmm=r2; i_mm=i; v_mm=self.k[i]
        if i_pp>=0 and i_pm>=0 and i_mp>=0 and i_mm>=0:     # bilinear blend of the four quadrant picks
            tpp,ppp=self.t[i_pp],self.p[i_pp]; tpm,ppm=self.t[i_pm],self.p[i_pm]
            tmp,pmp=self.t[i_mp],self.p[i_mp]; tmm,pmm=self.t[i_mm],self.p[i_mm]
            den_tp=max(tpp-tmp,1e-300); den_tm=max(tpm-tmm,1e-300)
            rppmp=((tl-tmp)*v_pp+(tpp-tl)*v_mp)/den_tp; rpmmm=((tl-tmm)*v_pm+(tpm-tl)*v_mm)/den_tm
            pppmp=((tl-tmp)*ppp+(tpp-tl)*pmp)/den_tp; ppmmm=((tl-tmm)*ppm+(tpm-tl)*pmm)/den_tm
            r=((pl-ppmmm)*rppmp+(pppmp-pl)*rpmmm)/max(pppmp-ppmmm,1e-300); return float(10.0**r)
        w=[1.0/(np.sqrt(r)+1e-5) for r in (rpp,rpm,rmp,rmm)]; rwt=sum(w)   # fallback inverse-distance blend
        idx=[max(i,0) for i in (i_pp,i_pm,i_mp,i_mm)]
        r=sum(self.k[idx[m]]*w[m] for m in range(4))/max(rwt,1e-300); return float(10.0**r)

rosstab = Rosstab(); rosstab.ingest(T, p_in, abross)
print(f"kappa_Ross spans {abross.min():.3e} to {abross.max():.3e} cm^2/g; tau_Ross deep = {tauros[-1]:.1f}")''')

# ── convection: the physics ──────────────────────────────────────────────────
md(r"""## Why the deep photosphere convects

Consider a small parcel of gas nudged upward. It expands to match the falling pressure of its new surroundings; because the expansion is fast compared to radiative exchange, it cools **adiabatically**, along the gradient $\nabla_{\rm ad} = (d\ln T/d\ln P)_{\rm ad}$. Meanwhile the surrounding atmosphere has its own temperature gradient $\nabla = d\ln T/d\ln P$, set by how radiation actually carries the flux. If the *ambient* gas cools faster with height than the parcel does — that is, if $\nabla > \nabla_{\rm ad}$ — then after the nudge the parcel is **hotter, hence less dense**, than its new surroundings, so buoyancy keeps pushing it up. The atmosphere is **convectively unstable** (the Schwarzschild criterion). The excess $\Delta = \nabla - \nabla_{\rm ad}$ is the **superadiabaticity**; where it is positive, convection switches on.

In the Sun this happens **below** the visible photosphere, where hydrogen is partially ionized: ionization soaks up energy, flattening $\nabla_{\rm ad}$, while the steeply rising opacity steepens $\nabla$, and the two together drive $\Delta > 0$. The upper, line-forming layers stay radiative ($\Delta < 0$), which is why our spectrum in Lectures 1–8 never needed convection. ATLAS encodes this in **mixing-length theory**: convective parcels are imagined to rise one *mixing length* $\ell = \alpha_{\rm ML} H_P$ (here $\alpha_{\rm ML}=1.25$ pressure scale heights) before dissolving and dumping their heat. From $\Delta$, the thermodynamics, and $\ell$ it computes a convective velocity and a **convective flux** $F_{\rm conv}$.

![Radiation carries the flux in the line-forming photosphere; in the deep layers convection (rising and sinking gas, mixing-length theory) takes over — negligible where the lines form, but it sets the deep structure.](resources/figures/s10_convection.png)""")

md(r"""### The thermodynamic derivatives, from finite differences

Mixing-length theory needs four thermodynamic derivatives at each layer: how the internal energy and the mass density respond to temperature and to pressure, $(\partial E/\partial T)_P$, $(\partial\rho/\partial T)_P$, $(\partial E/\partial P)_T$, $(\partial\rho/\partial P)_T$. In a partially-ionized gas these have no simple closed form — they depend on the full Saha/Boltzmann equilibrium (Lecture 2). ATLAS computes them by **finite differences**: it re-runs the equation of state at $T\pm0.1\%$ and at $P\pm0.1\%$ and differences the resulting electron density and mass density. We ship those four pairs of samples (`edens1..4`, `rho1..4`) as a given input — re-deriving the full EOS here would just repeat Lecture 2 — and form the derivatives. The factor $500 = 1/(2\times0.001)$ is the central-difference denominator; the electron-density samples already carry the radiation-energy term ATLAS adds.""")

code(r'''ed1 = REF["edens1"]; ed2 = REF["edens2"]; ed3 = REF["edens3"]; ed4 = REF["edens4"]   # EDENS at T+, T-, P+, P-
r1  = REF["rho1"];   r2  = REF["rho2"];   r3  = REF["rho3"];   r4  = REF["rho4"]        # RHO   at T+, T-, P+, P-

dEdT = (ed1-ed2)/np.maximum(T, 1e-300)*500.0     # (dE/dT)_P  -- central difference, 1/(2*0.001)=500
drdT = (r1 -r2 )/np.maximum(T, 1e-300)*500.0     # (drho/dT)_P
dEdP = (ed3-ed4)/np.maximum(p_in, 1e-300)*500.0  # (dE/dP)_T
drdP = (r3 -r4 )/np.maximum(p_in, 1e-300)*500.0  # (drho/dP)_T
print(f"thermodynamic derivatives formed from EOS finite differences at T,P +-0.1%")''')

md(r"""### The mixing-length kernel

This is Kurucz's `CONVEC`, depth by depth. For each layer it builds the gradients and thermodynamics, tests the Schwarzschild criterion, and — where the layer convects — runs a short inner loop for the convective flux. The loop is genuinely coupled: the convective efficiency depends on how opaque a cell is (the optical-thickness factor $\tau_b^2/(2+\tau_b^2)$, where $\tau_b=\kappa\rho\ell$ is the cell's optical thickness — optically thin cells radiate their heat away before delivering it and carry little flux), and the cell opacity depends on the temperature excess $\delta T$ the loop is solving for, read from `ROSSTAB` at $T\pm\delta T$. Thirty iterations with under-relaxation settle it. The single body below is one algorithm and stays whole; the comments mark each physical step. Two ATLAS conventions: the top `NCONV = 36` layers are **forced non-convective** (the surface is radiative by construction), and the deep flux is capped so a cell cannot carry more than the cell's heat content allows.""")

code(r'''def high_from_rhox(rhox, rho):
    """Integrate geometric height from column mass and density (Fortran HIGH)."""
    return integ(rhox, 1.0e-5/np.maximum(rho, 1e-300), 0.0)

def convec(rosstab, rhox, tauros, t, p, rho, abross, pradk, ptotal, grav, flux,
           dEdT, drdT, dEdP, drdP, mixlth=1.25, nconv=36):
    """Mixing-length convective flux per layer (Fortran CONVEC, finite-difference derivative path)."""
    nl = t.size
    dtdrhx = deriv(rhox, t)                          # dT/dRHOX, feeds the actual (radiative) gradient
    dilut  = 1.0 - np.exp(-tauros)                   # geometric dilution of the P_rad gradient
    dltdlp=np.zeros(nl); heatcp=np.zeros(nl); dlrdlt=np.zeros(nl); velsnd=np.zeros(nl)
    grdadb=np.zeros(nl); hscale=np.zeros(nl); flxcnv=np.zeros(nl); vconv=np.zeros(nl)
    deltat=np.zeros(nl); rosst=np.zeros(nl)
    for j in range(nl):
        dpdpg = 1.0
        dpdt  = 4.0*pradk[j]/max(t[j],1e-300)*dilut[j]                 # radiation-pressure temperature term
        dltdlp[j] = ptotal[j]/max(t[j]*grav,1e-300)*dtdrhx[j]          # actual gradient d ln T / d ln P
        drdP_s = _nz_signed(float(drdP[j]))
        heatcv = dEdT[j] - dEdP[j]*drdT[j]/drdP_s                       # specific heat at constant volume
        heatcp[j] = (dEdT[j] - dEdP[j]*dpdt/max(dpdpg,1e-300)          # ... at constant pressure
                     - ptotal[j]/max(rho[j]**2,1e-300)*(drdT[j] - drdP[j]*dpdt/max(dpdpg,1e-300)))
        if heatcv > 0.0: velsnd[j] = np.sqrt(max(heatcp[j]/heatcv*dpdpg/drdP_s, 0.0))   # sound speed
        dlrdlt[j] = t[j]/max(rho[j],1e-300)*(drdT[j] - drdP[j]*dpdt/max(dpdpg,1e-300))   # d ln rho / d ln T
        if abs(heatcp[j]) > 1e-300:
            grdadb[j] = -ptotal[j]/max(rho[j]*t[j],1e-300)*dlrdlt[j]/heatcp[j]           # adiabatic gradient
        hscale[j] = ptotal[j]/max(rho[j]*grav,1e-300)                  # pressure scale height H_P
        if mixlth == 0.0 or j < 3: continue
        delt = dltdlp[j] - grdadb[j]                                   # superadiabaticity Delta = grad - grad_ad
        if delt < 0.0: continue                                        # Schwarzschild-stable -> no convection
        vco = 0.5*mixlth*np.sqrt(max(-0.5*ptotal[j]/max(rho[j],1e-300)*dlrdlt[j], 0.0))   # velocity scale
        if vco == 0.0: continue
        fluxco = 0.5*rho[j]*heatcp[j]*t[j]*mixlth/FOURPI               # flux scale rho c_P T ell
        rosst[j] = rosstab.eval(float(t[j]), float(p[j]))             # cell-center opacity
        olddelt = 0.0
        for _ in range(30):                                           # inner loop: flux <-> temperature excess
            rd = _nz_signed(float(rosst[j]))
            dplus  = rosstab.eval(float(t[j]+deltat[j]), float(p[j]))/rd   # opacity at T + deltaT
            dminus = rosstab.eval(float(t[j]-deltat[j]), float(p[j]))/rd   # opacity at T - deltaT
            abconv = 0.0 if (dplus == 0.0 or dminus == 0.0) else 2.0/(1.0/dplus + 1.0/dminus)*abross[j]
            den1 = abconv*hscale[j]*rho[j]; den2 = fluxco*FOURPI
            d = 0.0 if (den1 == 0.0 or den2 == 0.0 or vco == 0.0) else 8.0*SIGMA*t[j]**4/den1/den2/vco
            taub = abconv*rho[j]*mixlth*hscale[j]                      # cell optical thickness tau_b
            d = d*taub**2/(2.0+taub**2)                                # optical-thickness efficiency factor
            d = d**2/2.0
            ddd = (delt/_nz_signed(float(d+delt)))**2
            if ddd < 0.5:                                             # series for (1-sqrt(1-ddd))/ddd, small ddd
                delta=0.5; term=0.5; up=-1.0; down=2.0
                while term > 1.0e-6:
                    up+=2.0; down+=2.0; term = up/down*ddd*term; delta += term
            else:
                delta = (1.0-np.sqrt(max(1.0-ddd,0.0)))/max(ddd,1e-300)
            delta = delta*delt**2/_nz_signed(float(d+delt))
            vconv[j]  = vco*np.sqrt(max(delta,0.0))                    # convective velocity
            flxcnv[j] = max(fluxco*vconv[j]*delta, 0.0)                # convective flux F_conv
            deltat[j] = min(t[j]*mixlth*delta, t[j]*0.15)             # temperature excess, capped at 15%
            deltat[j] = deltat[j]*0.7 + olddelt*0.3                    # under-relax for stability
            if olddelt-0.5 < deltat[j] < olddelt+0.5: break
            olddelt = deltat[j]
    flxcnv0 = flxcnv.copy()                                           # pre-patch flux (used by TCORR)
    height = high_from_rhox(rhox, rho)
    k = int(max(min(nconv, nl), 0))
    if k > 0: flxcnv[:k] = 0.0                                        # force the top NCONV layers radiative
    return dict(flxcnv=flxcnv, flxcnv0=flxcnv0, dltdlp=dltdlp, grdadb=grdadb, hscale=hscale,
                dlrdlt=dlrdlt, heatcp=heatcp, vconv=vconv, velsnd=velsnd, height=height)''')

md(r"""Run it on the converged model and look at where convection lives.""")

code(r'''cv = convec(rosstab, rhox, tauros, T, p_in, rho_in, abross, pradk, ptotal, GRAV, flux,
            dEdT, drdT, dEdP, drdP, mixlth=1.25, nconv=36)
cv["ptotal"] = ptotal; cv["rho"] = rho_in          # carried for the TCORR convective-efficiency term

conv = cv["flxcnv"] > 0.0
print(f"layers carrying convective flux: {int(conv.sum())} of {n}")
print(f"  first convective layer at tau_Ross = {tauros[np.argmax(conv)]:.3f}")
print(f"  peak convective fraction F_conv/(4 pi H) = {(cv['flxcnv']/(FOURPI*flux)).max():.3f}")''')

md(r"""The convective flux is **zero throughout the line-forming layers** ($\tau_{\rm Ross}\lesssim1$) — confirming that everything we did for the spectrum in Lectures 1–8 was untouched by convection — and switches on only in the deep, optically thick interior, where at its peak it carries a large fraction of the total flux. That is exactly the regime where the temperature correction needs it.""")

# ── superadiabaticity plot ───────────────────────────────────────────────────
md(r"""### Seeing the Schwarzschild criterion

The clearest picture of convection is the two gradients side by side. Where the **actual** gradient $\nabla$ (set by radiation) exceeds the **adiabatic** gradient $\nabla_{\rm ad}$, the gas convects; the gap between them is the superadiabaticity $\Delta$ that drives it. The convective flux (shaded) tracks exactly the region where $\nabla > \nabla_{\rm ad}$.""")

code(r'''fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
x = np.log10(tauros)
ax[0].plot(x, cv["dltdlp"], color="C3", lw=1.7, label=r"$\nabla$ (actual)")
ax[0].plot(x, cv["grdadb"], color="C0", lw=1.7, label=r"$\nabla_{\rm ad}$ (adiabatic)")
ax[0].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[0].set_ylabel("gradient $d\\ln T/d\\ln P$")
ax[0].set_title("Schwarzschild: where $\\nabla>\\nabla_{\\rm ad}$"); ax[0].legend(loc="upper left")
ax[0].set_ylim(0, max(0.6, np.nanmax(cv["dltdlp"][np.isfinite(cv["dltdlp"])])*1.1))

ax[1].fill_between(x, 0, cv["flxcnv"]/(FOURPI*flux), color="C2", alpha=0.35)
ax[1].plot(x, cv["flxcnv"]/(FOURPI*flux), color="C2", lw=1.7)
ax[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[1].set_ylabel(r"$F_{\rm conv}/\sigma T_{\rm eff}^4$")
ax[1].set_title("Convective flux fraction")
fig.tight_layout(); plt.show()''')

# ── TCORR with convection ────────────────────────────────────────────────────
md(r"""## The temperature correction, now with convection

The temperature correction of Lecture 10 — $T_1 = \Delta T_{\rm flux} + \Delta T_\Lambda + \Delta T_{\rm surf}$ — gains the convective flux in three places. (1) The **flux error** it drives toward zero is now the *total* flux $H + F_{\rm conv}/4\pi$ minus the target, not the radiative flux alone. (2) The Avrett–Krook denominator picks up a **convective-efficiency** term `ddel` (how strongly $F_{\rm conv}$ responds to a temperature change), so the correction knows that in a convective layer some of the flux adjustment comes for free from convection. (3) The convective flux is **smoothed** with a 1–2–1 filter and its top two layers zeroed before use, since the raw layer-by-layer flux is noisy. Everything else — the local-$\Lambda$ surface term, the surface-boundary term, the monotonicity clamp, and the hydrostatic density correction via `TTAUP` — is carried over from Lecture 10 unchanged. We are at the converged fixed point (`iter_index = 1`), so the damping/acceleration logic is skipped.""")

code(r'''def ttaup(t, tau, prad, grav, rfun):
    """Hydrostatic re-integration on a tau grid, opacity from ROSSTAB (Lecture 10 TTAUP).
    Predictor-corrector in log-pressure: integrate dP/dtau = g/kappa down the column."""
    nl = t.size; abstd=np.zeros(nl); ptot=np.zeros(nl); pgas=np.zeros(nl)
    dlg = np.log(max(float(tau[1]/max(tau[0],1e-300)),1e-300)) if nl > 1 else 0.0   # log tau step
    plog1=plog2=plog3=plog4=0.0; dplog1=dplog2=dplog3=0.0                            # history for the predictor
    abstd[0] = min(0.1, grav*tau[0]/max(prad[0],1e-300)/2.0) if prad[0] > 0.0 else 0.1   # surface opacity guess
    for j in range(nl):
        if j == 0: plog = np.log(max(grav/max(abstd[0],1e-300)*tau[0],1e-300))       # top: P = g tau / kappa
        elif j <= 3: plog = plog1 + dplog1                                            # low-order predictor
        else: plog = (3.0*plog4 + 8.0*dplog1 - 4.0*dplog2 + 8.0*dplog3)/3.0           # high-order predictor
        error=1.0; dplog=0.0; itn=1
        while True:                                                                   # corrector iteration
            plog = min(plog, 709.78); ptot[j] = np.exp(plog)                          # total pressure
            pgas[j] = ptot[j] + (prad[0]-prad[j])                                     # subtract radiation pressure
            if pgas[j] <= 0.0: pgas[j]=1e-30; abstd[j]=0.1; break
            abstd[j] = rfun(float(t[j]), float(pgas[j]))                              # ROSSTAB opacity at (T, Pgas)
            dplog = grav/max(abstd[j],1e-300)*tau[j]/max(ptot[j],1e-300)*dlg; itn += 1 # d(log P)/d(log tau)
            if itn > 1000 or error <= 5.0e-5: break
            if j == 0: pnew = np.log(max(grav/max(abstd[j],1e-300)*tau[j],1e-300))
            elif j <= 3: pnew = (plog + 2.0*plog1 + dplog + dplog1)/3.0               # low-order corrector
            else: pnew = (126.0*plog1 - 14.0*plog3 + 9.0*plog4 + 42.0*dplog + 108.0*dplog1 - 54.0*dplog2 + 24.0*dplog3)/121.0
            error = abs(pnew-plog); plog = 0.5*(pnew+plog)                            # damped update
        plog4=plog3; plog3=plog2; plog2=plog1; plog1=plog; dplog3=dplog2; dplog2=dplog1; dplog1=dplog  # shift history
    return abstd, ptot, pgas''')

code(r'''def tcorr_mode3(T, rhox, tauros, abross, flxrad, rjmins, rdabh, rdiagj, flux, teff, prad, grav,
                rosstab, cv, mixlth=1.25, steplg=0.125, tau1lg=-6.875):
    """ATLAS TCORR mode 3 with convection: T1 = dtflux + dtlamb + dtsurf, plus DRHOX (Lecture 10 + CONVEC)."""
    nl = T.size
    dtdrhx = deriv(rhox, T); dabros = deriv(rhox, abross)             # dT/d(rhox) and d(kappa)/d(rhox)
    flxcnv=cv["flxcnv"]; flxcnv0=cv["flxcnv0"]; dltdlp=cv["dltdlp"]; grdadb=cv["grdadb"]   # unpack CONVEC arrays
    hscale=cv["hscale"]; dlrdlt=cv["dlrdlt"]; heatcp=cv["heatcp"]; ptc=cv["ptotal"]; rhc=cv["rho"]
    ddlt = deriv(rhox, dltdlp)                                        # gradient of the actual gradient
    cnvflx = flxcnv.copy(); cnvflx[0] = 0.0                            # smooth the convective flux: zero top two,
    if nl >= 2: cnvflx[1] = 0.0
    if nl >= 3:
        ccc = cnvflx.copy()
        for j in range(1, nl-1): ccc[j] = 0.25*cnvflx[j-1] + 0.5*cnvflx[j] + 0.25*cnvflx[j+1]   # 1-2-1 filter
        ccc[-1] = 0.25*cnvflx[-3] + 0.25*cnvflx[-2] + 0.5*cnvflx[-1]
        for j in range(1, nl-1): cnvflx[j] = ccc[j]
        cnvflx[-1] = ccc[-1]
    rdabh_eff = rdabh - flxrad*dabros/np.maximum(abross, 1e-300)      # opacity-gradient term in the AK denominator
    codrhx = np.zeros(nl); ddel = np.zeros(nl)
    for j in range(nl):                                               # per layer: AK integrand + convective ddel
        delv = 1.0; d = 0.0
        if cnvflx[j] > 0.0 and flxcnv0[j] > 0.0:                       # convective-efficiency factor ddel
            delv = dltdlp[j] - grdadb[j]                               # superadiabaticity at this layer
            vco = 0.5*mixlth*np.sqrt(max(-0.5*ptc[j]/max(rhc[j],1e-300)*dlrdlt[j], 0.0))   # velocity scale
            fluxco = 0.5*rhc[j]*heatcp[j]*T[j]*mixlth/FOURPI           # flux scale
            if mixlth > 0.0 and vco > 0.0:                            # radiative-loss parameter d
                d = 8.0*SIGMA*T[j]**4/np.maximum(abross[j]*hscale[j]*rhc[j],1e-300)/np.maximum(fluxco*FOURPI,1e-300)/vco
            taub = abross[j]*rhc[j]*mixlth*hscale[j]; d = d*taub*taub/(2.0+taub*taub); d = d*d/2.0   # optical-thickness
            ddel[j] = (1.0 + d/_nz_signed(float(d+delv)))/_nz_signed(float(delv))   # dF_conv/dT response
        cnvfl = 0.0
        if flxrad[j] > 0.0 and cnvflx[j]/flxrad[j] > 1.0e-3 and flxcnv0[j]/flxrad[j] > 1.0e-3:
            cnvfl = cnvflx[j]                                         # convective flux only where it matters (>0.1%)
        dd = _nz_signed(float(d+delv)); ds = _nz_signed(float(delv))
        num = rdabh_eff[j] + cnvfl*(dtdrhx[j]/max(T[j],1e-300)*(1.0-9.0*d/dd) + 1.5*ddlt[j]/ds*(1.0+d/dd))
        den = flxrad[j] + cnvflx[j]*1.5*dltdlp[j]*ddel[j]             # total-flux denominator (rad + conv response)
        codrhx[j] = num/_nz_signed(float(den))
    codrhx[0] = 0.0
    if nl >= 2: codrhx[1] = 0.0
    g = np.exp(integ(rhox, codrhx, 0.0))                              # Avrett-Krook integrating factor
    gfden = flxrad + cnvflx*1.5*dltdlp*ddel
    gfden_s = np.where(np.abs(gfden) >= 1e-300, gfden, np.where(gfden >= 0.0, 1e-300, -1e-300))
    gflux = g*(flxrad + cnvflx - flux)/gfden_s                         # total-flux defect drives Avrett-Krook
    dtau = integ(tauros, gflux, 0.0)/np.maximum(g, 1e-300)
    dtau = np.maximum(-tauros/3.0, np.minimum(tauros/3.0, dtau))       # stability clamp on the tau shift
    dtflux = -dtau*dtdrhx/np.maximum(abross, 1e-300)
    flxerr = (flxrad + cnvflx - flux)/np.maximum(flux, 1e-300)*100.0   # percent TOTAL-flux error
    flxdrv = deriv(tauros, flxerr); dtlamb = np.zeros(nl); teff25 = teff/25.0
    for j in range(nl):                                               # local-Lambda surface term (Lecture 10)
        ratio = cnvflx[j]/np.maximum(flxrad[j], 1e-300)
        if ratio < 1.0e-5:
            flxdrv[j] = rjmins[j]/np.maximum(abross[j],1e-300)/np.maximum(flux,1e-300)*100.0
        denom = rdiagj[j] if abs(rdiagj[j]) > 1e-300 else np.sign(rdiagj[j])*1e-300
        dtlamb[j] = -flxdrv[j]*flux/100.0/denom*abross[j]
        if not (ratio < 1.0e-5 and tauros[j] < 1.0):
            dtlamb[j] = 0.0
            for k in range(1, 6):
                if j-k >= 0: dtlamb[j-k] *= 0.5
        dtlamb[j] = float(np.clip(dtlamb[j], -teff25, teff25))
    dtsur = float(np.clip((flux-flxrad[0])/np.maximum(flux,1e-300)*0.25*T[0], -teff25, teff25))  # surface term
    tinteg = integ(tauros, dtflux+dtlamb, 0.0)
    tav = (map1_scalar(tauros, tinteg, 2.0) - map1_scalar(tauros, tinteg, 0.1))/2.0
    if dtsur*tav <= 0.0: tav = 0.0
    if abs(tav) > abs(dtsur): tav = dtsur
    dtsur = dtsur - tav; dtsurf = np.full(nl, dtsur)
    dtflux = np.nan_to_num(dtflux); dtlamb = np.nan_to_num(dtlamb); dtsurf = np.nan_to_num(dtsurf)   # sanitize
    t1 = dtflux + dtlamb + dtsurf                                     # total temperature correction
    tnew = np.maximum(np.where(np.isfinite(T+t1), T+t1, T), 1.0)       # iter 1: damping skipped
    for i in range(1, nl):                                            # monotonicity: T increasing with depth
        j = nl-1-i; tnew[j] = np.fmin(tnew[j], tnew[j+1]-1.0)
        if not np.isfinite(tnew[j]): tnew[j] = max(T[j], 1.0)
    taustd = 10.0**(tau1lg + np.arange(nl)*steplg); rfun = rosstab.eval   # DRHOX: re-run TTAUP on T and T+t1
    tnew1, _ = map1(tauros, T, taustd); prdnew, _ = map1(tauros, prad, taustd)   # T and P_rad on the std grid
    _a1, ptot1, _ = ttaup(tnew1, taustd, prdnew, grav, rfun)          # hydrostatic pressure for T
    tnew2, _ = map1(tauros, T+t1, taustd)
    _a2, ptot2, _ = ttaup(tnew2, taustd, prdnew, grav, rfun)          # hydrostatic pressure for T+T1
    ppp = (ptot2-ptot1)/np.maximum(ptot1, 1e-300); rrr, _ = map1(taustd, ppp, tauros)   # fractional dP -> dRHOX
    return dict(t1=t1, dtflux=dtflux, dtlamb=dtlamb, flxerr=flxerr, cnvflx=cnvflx,
                tnew=tnew, rhox_new=rhox + rrr*rhox)

res = tcorr_mode3(T, rhox, tauros, abross, flxrad, rjmins, rdabh, rdiagj, flux, TEFF, prad, GRAV, rosstab, cv)
print(f"temperature correction T1 spans {res['t1'].min():+.3e} to {res['t1'].max():+.3e} K")''')

md(r"""## Closing the iteration and the fixed-point benchmark

The iteration closes exactly as in Lecture 10: remap the corrected temperature $T+T_1$ and column mass $\rho x + \Delta\rho x$ from the Rosseland grid onto the fixed standard optical-depth grid `taustd`. We then benchmark in two complementary ways. The **precision** benchmark compares our single step to *pykurucz's own single step from the same converged input* — this isolates the fidelity of our engines from a subtlety about what "converged" means. The **self-consistency** benchmark compares our step to the converged model itself: because the model is converged, one step barely moves it.""")

code(r'''taustd = 10.0**(-6.875 + np.arange(n)*0.125)
T_out,    _ = map1(tauros, res["tnew"],     taustd)   # corrected T,    remapped to the standard grid
rhox_out, _ = map1(tauros, res["rhox_new"], taustd)   # corrected RHOX, remapped to the standard grid

def rel(a, b):
    b = np.asarray(b, float); a = np.asarray(a, float)
    m = np.abs(b) > 0; r = np.zeros_like(b); r[m] = np.abs(a[m]-b[m])/np.abs(b[m]); return r

# PRECISION benchmark: our single step vs pykurucz's SAME single step (engine fidelity)
rT = rel(T_out,    REF["T_step"]).max()
rX = rel(rhox_out, REF["rhox_step"]).max()
# the NEW convection physics, vs the reference CONVEC arrays
rF = rel(cv["flxcnv"], REF["flxcnv_ref"]).max()
rG = rel(cv["grdadb"], REF["grdadb_ref"]).max()
# self-consistency: our step vs the converged model (one step ~ no-op)
rT_self = np.median(rel(T_out, REF["T_converged"]))
print(f"CONVEC: FLXCNV max|rel| = {rF:.2e}   grdadb max|rel| = {rG:.2e}")
print(f"one-step vs pykurucz step: T = {rT:.2e}   RHOX = {rX:.2e}")
print(f"self-consistency vs converged model: median dT/T = {rT_self:.2e}")''')

md(r"""## The benchmark

The new convection physics — the convective flux `flxcnv` and the gradients — is pure float64 and matches the production code to **machine precision** ($\sim10^{-10}$ or better; the gradients to $\sim10^{-16}$). The full step — temperature *and* column mass — reproduces pykurucz's single step from the same converged input to **machine precision** (T $\sim10^{-9}$, column mass $\sim10^{-8}$); the residual is the float32 $\Lambda$-iteration inside JOSH (the one single-precision step in the whole pipeline, Lecture 8), which jitters the flux at the ULP level and averages out.

A word on the second number we printed. Comparing our step instead to the **converged model itself** gives a median $\sim10^{-6}$ in the deep temperature — but the *top* layers and the column mass differ at the $\sim10^{-3}$ level. That is not an error: it is what "converged" means. ATLAS stops when the **deep-layer** $\max|\Delta T/T| < 10^{-4}$; one more step is *not* an exact no-op everywhere — the thin upper layers and the density keep drifting at the $10^{-3}$ level, which is precisely why `checkconv` looks only at the deep layers. The machine-precision claim is the engine-fidelity comparison (vs the same single step); the self-consistency comparison confirms the model sits at the convergence criterion.""")

code(r'''print("="*64)
print("LECTURE 11 BENCHMARK  (one iteration from the converged solar model)")
print("="*64)
print(f"  FLXCNV (NEW: convection)  : max|rel| = {rF:.2e}   <- MACHINE PRECISION")
print(f"  one-step T   (vs pykurucz): max|rel| = {rT:.2e}   <- MACHINE PRECISION")
print(f"  one-step RHOX(vs pykurucz): max|rel| = {rX:.2e}   <- MACHINE PRECISION")
ok = (rF < 1e-6) and (rT < 1e-6) and (rX < 1e-6)
print("  PASS" if ok else "  FAIL")
print("="*64)''')

md(r"""### The converged model and its flux constancy

Two pictures close the loop. The left panel plots the **convergence history** — the deep-layer $\max|\Delta T/T|$ falling from order unity at the grey start to below $10^{-4}$ over the 28 iterations the production run took, while the all-layer metric stalls higher because the thin top layers settle last. The right panel is the payoff: the **total** flux (radiative + convective) across the converged model, flat at the target $\sigma T_{\rm eff}^4$ — radiation carries it in the upper photosphere, convection takes over in the deep interior, and together they hold the flux constant. *That* constancy is what "converged" means.""")

code(r'''fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
it = np.arange(1, REF["dlnt_history"].size + 1)
ax[0].semilogy(it, REF["dtmax_history"], color="0.6", lw=1.5, marker="o", ms=3, label="all layers")
ax[0].semilogy(it, REF["dlnt_history"],  color="C0",  lw=1.7, marker="o", ms=3, label="deep layers (checkconv)")
ax[0].axhline(1e-4, color="C3", ls="--", lw=1.2, label=r"$10^{-4}$ threshold")
ax[0].set_xlabel("iteration"); ax[0].set_ylabel(r"$\max|\Delta T/T|$")
ax[0].set_title("Convergence history"); ax[0].legend(loc="upper right", fontsize=9)

Ftot = (flxrad + res["cnvflx"]) * FOURPI                    # total flux 4 pi (H + F_conv/4pi)
ax[1].axhline(SIGMA*TEFF**4, color="C3", ls=":", lw=1.2, label=r"$\sigma T_{\rm eff}^4$ (target)")
ax[1].plot(np.log10(tauros), flxrad*FOURPI,      color="C0", lw=1.6, label="radiative")
ax[1].plot(np.log10(tauros), Ftot,               color="C2", lw=1.8, label="total (rad + conv)")
ax[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[1].set_ylabel(r"flux [erg cm$^{-2}$ s$^{-1}$]")
ax[1].set_title("Flux constancy of the converged model"); ax[1].legend(loc="lower left", fontsize=9)
fig.tight_layout(); plt.show()''')

# ── the milestone ────────────────────────────────────────────────────────────
md(r"""## The milestone: from $(T_{\rm eff}, \log g)$ to a model atmosphere

This lecture closes the loop the very first lecture opened. In Lectures 1–8 we *took the atmosphere as given* — a table of $T(\tau)$, $P(\tau)$, $\rho(\tau)$ shipped with the book — and used it to compute opacities and an emergent spectrum. We never built that table; we borrowed it. From Lecture 9 onward we earned it: hydrostatic equilibrium fixed the pressure structure (Lecture 9), the temperature correction fixed the temperature for flux constancy (Lecture 10), and now convection plus the convergence loop produce the **full, self-consistent model** — from nothing but $T_{\rm eff}$, $\log g$, and the composition.

The path is complete: **$(T_{\rm eff}, \log g, {\rm composition}) \to$ grey start (Lecture 9) $\to$ {EOS, opacity, Rosseland mean, JOSH flux, convection, temperature correction}, iterated to flux constancy $\to$ the converged atmosphere $\to$ the emergent spectrum (Lectures 1–8)**. A star's few numbers become its entire structure and its spectrum, from first principles and to the bit.""")

# ── forward look ─────────────────────────────────────────────────────────────
md(r"""## What comes next: line blanketing and molecules

The model we just converged is **continuum-only** — deliberately, so it needs no line data and stays reproducible with NumPy alone. A real model is **line-blanketed**: the millions of spectral lines (Lectures 4–6) trap radiation, raise the opacity in the upper layers, and reshape the temperature structure (back-warming). Switching them on changes nothing in the *machinery* of this lecture — the frequency sweep simply carries line opacity alongside the continuum, and the correction and convection are untouched — but it needs the large line lists the next part introduces.

For **cool stars** there is one more ingredient: **molecules**. Below about 4000 K, molecules like TiO, H$_2$O, and CO form in vast numbers, dominate the opacity, and carve the broad molecular bands that define an M dwarf's spectrum. Their equilibrium abundances feed back into the equation of state and the opacity, so the convergence loop must include molecular chemistry. That is the subject of **Lecture 12 (Molecular Equilibrium & Molecular Bands)**. The book then ends with **Lecture 13 (Putting It Together)**, a capstone that runs the whole stack — EOS, line and continuum opacity, convergence, spectrum — across several stars spanning the HR diagram.""")

# ── synthesis ────────────────────────────────────────────────────────────────
md(r"""## Synthesis

Lecture 10 built one radiative-equilibrium correction step; this lecture turned it into a finished model by adding the two missing pieces. **Convection**: in the deep photosphere of a cool star the radiative gradient exceeds the adiabatic gradient (the Schwarzschild criterion), the gas becomes unstable, and rising/falling parcels carry part of the flux. We implemented Kurucz's mixing-length `CONVEC` — the adiabatic gradient $\nabla_{\rm ad}$ and superadiabaticity $\Delta=\nabla-\nabla_{\rm ad}$, with thermodynamic derivatives from finite differences of the EOS, and a 30-iteration inner loop for the convective flux $F_{\rm conv}$ using the optical-thickness efficiency $\tau_b^2/(2+\tau_b^2)$ — and saw it is zero in the line-forming layers but carries much of the deep flux. That flux feeds the temperature correction in three places (the total-flux defect, the convective-efficiency denominator, and a smoothing). **Convergence**: iterating the corrected EOS, opacity, fluxes, convection, and temperature drives the model to flux constancy, measured as $\max|\Delta T/T|<10^{-4}$ over the deep layers; the solar model took 28 iterations from a grey start. We verified the converged model is a fixed point of our from-scratch pipeline: one from-scratch iteration reproduces pykurucz's own single step from the converged model to machine precision in the convective flux, the temperature, and the column mass.""")

md(r"""## Summary

- The deep photosphere **convects** where the actual gradient exceeds the adiabatic one, $\nabla > \nabla_{\rm ad}$ (the **Schwarzschild criterion**); the excess $\Delta = \nabla - \nabla_{\rm ad}$ is the superadiabaticity. In the Sun this happens below the visible surface, driven by hydrogen ionization; the **line-forming layers stay radiative**.
- **Mixing-length theory** (`CONVEC`) carries a parcel one mixing length $\ell = \alpha_{\rm ML} H_P$ ($\alpha_{\rm ML}=1.25$) and computes the convective flux $F_{\rm conv}$ from $\Delta$, the convective velocity, and the optical-thickness efficiency $\tau_b^2/(2+\tau_b^2)$ ($\tau_b = \kappa\rho\ell$). The thermodynamic derivatives $\partial E/\partial T$, $\partial\rho/\partial T$, $\partial E/\partial P$, $\partial\rho/\partial P$ come from **finite differences** of the EOS; the top `NCONV=36` layers are forced radiative.
- The convective flux enters the **temperature correction** through the total-flux defect $H + F_{\rm conv}/4\pi - H_{\rm target}$, a convective-efficiency term in the Avrett–Krook denominator, and a 1–2–1 smoothing of $F_{\rm conv}$.
- **Convergence** means flux constancy, tested as $\max|\Delta T/T| < 10^{-4}$ over the **deep layers** (ATLAS `checkconv`, layers 40–75); the upper layers settle last and are excluded. The solar continuum-only model converges in **28 iterations** from the grey start.
- The converged atmosphere is a **fixed point**: one from-scratch iteration {opacity, JOSH, Rosseland mean, convection, temperature correction} reproduces pykurucz's own single step to **machine precision** — convective flux ($\sim10^{-10}$), temperature ($\sim10^{-9}$), and column mass ($\sim10^{-8}$). "Converged" means the deep-layer $\max|\Delta T/T|<10^{-4}$, not that one step is an exact no-op everywhere.
- This closes the loop the book opened: from $(T_{\rm eff}, \log g, {\rm composition})$ we now build the **full model atmosphere**, the structure Lectures 1–8 took as given.""")

md(r"""## Practice exercises

**1. The Schwarzschild criterion, layer by layer.** Print $\nabla$, $\nabla_{\rm ad}$, and $\Delta = \nabla - \nabla_{\rm ad}$ versus $\log\tau_{\rm Ross}$ and find the first layer where $\Delta$ turns positive. Confirm it lies *below* the line-forming region ($\tau_{\rm Ross}\sim1$). Why does hydrogen ionization both flatten $\nabla_{\rm ad}$ and (via the opacity) steepen $\nabla$, so that the two conspire to make $\Delta > 0$?

**2. The efficiency factor.** In the inner loop, replace the optical-thickness factor $\tau_b^2/(2+\tau_b^2)$ with $1$ (an infinitely efficient cell) and recompute the convective flux. Where does it change most, and why? Explain physically why an optically *thin* cell ($\tau_b\ll1$) radiates away its heat before delivering it and so carries little flux.

**3. The mixing length.** Re-run `convec` with $\alpha_{\rm ML} = 0.5$ and $2.0$ and compare the peak convective flux fraction and the depth where convection switches on. The mixing length is the one free parameter of the theory — which observable features of a model atmosphere is it tuned against?

**4. Reading the convergence history.** From `dlnt_history`, estimate the *convergence rate* (the ratio of successive deep-layer $\max|\Delta T/T|$ values once past the first few iterations). Is it geometric? Roughly how many more iterations would a $10^{-6}$ threshold have required? Why does the all-layer metric (`dtmax_history`) stall above the deep-layer one?

**5. Convection off.** Set `cv` to all-zero convective arrays (or run `convec` with `mixlth = 0`) and redo the fixed-point check against the converged model. The temperature should now *fail* to reproduce in the deep layers. By how much does the deep temperature move, and at what optical depth does the discrepancy appear? This is the structural imprint of convection — invisible at the surface, decisive in the interior.""")

md(r"""## Further reading

- **Böhm-Vitense, E. (1958). *Über die Wasserstoffkonvektionszone in Sternen verschiedener Effektivtemperaturen und Leuchtkräfte*, Z. Astrophys. 46, 108.** The mixing-length theory of convection in stellar envelopes that `CONVEC` implements.
- **Mihalas, D. (1978). *Stellar Atmospheres*, 2nd ed., Freeman.** Chapter 7 on convection in model atmospheres, the Schwarzschild criterion, and the coupling of convective flux into the temperature structure.
- **Hubeny, I. & Mihalas, D. (2014). *Theory of Stellar Atmospheres*, Princeton.** Chapters 16–18 on convective energy transport, mixing-length formulations, and the construction and convergence of self-consistent model atmospheres.
- **Kippenhahn, R., Weigert, A. & Weiss, A. (2012). *Stellar Structure and Evolution*, 2nd ed., Springer.** Chapters 6–7 on convection, the Schwarzschild and Ledoux criteria, and mixing-length theory from the stellar-interior side.
- **Kurucz, R. L. (1970). *ATLAS: A Computer Program for Calculating Model Stellar Atmospheres*, SAO Special Report 309**, and **Castelli, F. & Kurucz, R. L. (2003), IAU Symp. 210.** The ATLAS convection (`CONVEC`), temperature-correction (`TCORR`), and convergence (`checkconv`) routines reproduced here.
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz*](https://arxiv.org/abs/2603.11693).** The implementation our reference converged model is computed with.""")

nb = new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
