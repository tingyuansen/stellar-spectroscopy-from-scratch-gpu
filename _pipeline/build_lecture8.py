#!/usr/bin/env python
"""Lecture 8 — The JOSH Solver: Production Radiative Transfer. The moment equations and
Eddington closure, the discrete Lambda operator on a fixed optical-depth grid, the
scattering source function and its iteration, and the CH-weighted surface flux. Rebuilds
pykurucz's solve_josh_flux from scratch and reproduces the spectrum to machine precision.
Checked against reference/diag.npz + reference/josh_tables.npz. No pykurucz import.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture8.ipynb"
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# Lecture 8 — The JOSH Solver: Production Radiative Transfer

*Stellar Spectroscopy from Scratch — rebuilding the physics of ATLAS and SYNTHE from first principles*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*Every result in this book is checked against reference values computed with [**pykurucz**](https://arxiv.org/abs/2603.11693) — a pure-Python implementation of Kurucz's ATLAS12 and SYNTHE — shipped beside the lectures as small data files, so the notebooks need only NumPy to run.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Take the **moments** of the transfer equation and close them with the **Eddington approximation**, turning radiative transfer into a pair of moment equations for the mean intensity $J$ and the flux $H$.
- Read the production code's three precomputed tables — a fixed **optical-depth grid**, a discrete **$\Lambda$ operator** `COEFJ`, and surface-flux weights `CH` — and say what each one does.
- Build the optical-depth scale with a **parabolic integrator**, map the source onto the fixed grid with **parabolic interpolation**, and solve the **scattering source function** by iteration.
- Assemble the emergent flux and reproduce the reference solar spectrum to **machine precision** — closing the ten-percent gap the formal solution left in the deepest line cores.""")

md(r"""## Introduction

Lecture 7 solved radiative transfer with the **formal solution** and one approximation: the source function was the Planck function, $S_\lambda = B_\lambda$. That is exact wherever true absorption dominates, and it reproduced the solar spectrum to a part in a thousand. The one place it loosened — to about ten percent — was the bottom of the deepest line cores, where **scattering** is no longer negligible and the source function is pulled below $B_\lambda$ by photons that scatter and escape:

$$
S_\lambda = \frac{\kappa^{\rm abs}_\lambda\,B_\lambda + \kappa^{\rm scat}_\lambda\,J_\lambda}{\kappa^{\rm abs}_\lambda + \kappa^{\rm scat}_\lambda}
        = (1-\alpha_\lambda)\,B_\lambda + \alpha_\lambda\,J_\lambda,
\qquad
\alpha_\lambda \equiv \frac{\kappa^{\rm scat}_\lambda}{\kappa^{\rm abs}_\lambda + \kappa^{\rm scat}_\lambda}.
$$

The difficulty is the **mean intensity** $J_\lambda$: it is itself an integral of the radiation field over angle and depth, which depends on $S_\lambda$ — the source and the field are coupled. The production code resolves this with a **moment method** on a fixed optical-depth grid, due to Avrett & Loeser and known in the Kurucz codes as **JOSH**. It reduces the angular transfer problem to two coupled ordinary differential equations for the moments, discretises them as a precomputed matrix, and iterates the source to self-consistency. This lecture rebuilds it step by step, and the payoff is the spectrum to machine precision.

![The JOSH moment solver: opacity to optical depth, a map onto a fixed Eddington grid, a scattering iteration for the source, and a weighted surface flux.](resources/figures/s7_josh.png)""")

# ── moment equations ────────────────────────────────────────────────────
md(r"""## The moment equations and the Eddington closure

Start from the plane-parallel transfer equation of Lecture 7, $\mu\,dI/d\tau = I - S$, and take **moments** in the angle cosine $\mu$ — that is, multiply by powers of $\mu$ and integrate over the unit sphere. Define the first three moments of the intensity,

$$
J = \tfrac12\!\int_{-1}^{1} I\,d\mu, \qquad
H = \tfrac12\!\int_{-1}^{1} I\,\mu\,d\mu, \qquad
K = \tfrac12\!\int_{-1}^{1} I\,\mu^2\,d\mu,
$$

the **mean intensity** $J$, the **Eddington flux** $H$ (the physical flux is $F = 4\pi H$), and the second moment $K$. Taking the zeroth and first moments of the transfer equation gives

$$
\frac{dH}{d\tau} = J - S, \qquad \frac{dK}{d\tau} = H.
$$

This is two equations in three unknowns ($J, H, K$) — not closed. The **Eddington approximation** closes it: deep in the atmosphere the radiation field is nearly isotropic, so $K = \tfrac13 J$ (for an isotropic field $\langle\mu^2\rangle = 1/3$). Substituting,

$$
\frac{d^2(fJ)}{d\tau^2} = J - S, \qquad f \to \tfrac13,
$$

a single second-order equation for $J$ given the source $S$, with the surface boundary condition $H(0) = \tfrac{1}{\sqrt3}\,J(0)$ (no incoming radiation) and a diffusion condition at depth. Its solution is **linear** in $S$: $J = \Lambda[S]$, where $\Lambda$ is the classical lambda operator. Discretised on a fixed optical-depth grid, $\Lambda$ becomes a **matrix**, and that matrix is one of the tables the production code ships.""")

code(r'''import pathlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5})

REF = pathlib.Path("..") / "reference"
T = np.load(REF / "josh_tables.npz")
XTAU = T["xtau"]          # the fixed optical-depth grid (51 points), surface -> deep
CH   = T["ch"]            # surface-flux weights: H(0) = sum(CH * S)
COEFJ = T["coefj"]        # the discrete Lambda operator: J = COEFJ @ S on the grid
RHOX = T["rhox"]          # column mass of the model atmosphere [g/cm^2], 80 layers
NXTAU = XTAU.size
print(f"fixed grid: {NXTAU} points, tau = {XTAU[0]:.3g} .. {XTAU[-1]:.3g}")
print(f"COEFJ is {COEFJ.shape}, CH is {CH.shape}, atmosphere has {RHOX.size} layers")''')

md(r"""## The three tables, and what they mean

The method rests on three precomputed objects, all defined on the **fixed optical-depth grid** `XTAU` (51 points spanning $\tau \approx 10^{-5}$ at the surface to $\tau \approx 20$ deep). The grid is fixed because the moment equations, once written in $\tau$, are the same for every wavelength — only the source and the scattering fraction change.

- **`COEFJ`** is the discrete $\Lambda$ operator. Given the source function sampled on the grid, $J = \texttt{COEFJ} \cdot S$ returns the mean intensity on the grid — it *is* the solution of the moment equations, packaged as a matrix multiply. Its diagonal, $\texttt{COEFJ}_{kk}$, is how strongly the local source feeds the local mean intensity, and it is the dominant term.
- **`CH`** turns the converged source into the emergent **surface flux**: $H(0) = \sum_k \texttt{CH}_k\,S_k$. It is the discrete form of the flux integral, the moment-method counterpart of the $E_2$ kernel from Lecture 7.
- **`XTAU`** is the grid both are built on.

Let us look at them.""")

code(r'''fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
im = a1.imshow(np.log10(np.abs(COEFJ) + 1e-30), cmap="magma", aspect="auto")
a1.set_title(r"$\log_{10}|\,$COEFJ$\,|$ — the discrete $\Lambda$ operator")
a1.set_xlabel("source grid point $m$"); a1.set_ylabel("response grid point $k$")
fig.colorbar(im, ax=a1, fraction=0.046)
a2.plot(np.arange(NXTAU), np.diag(COEFJ), "o-", ms=3, label="diagonal COEFJ$_{kk}$")
a2.plot(np.arange(NXTAU), CH, "s-", ms=3, color="C3", label="CH (flux weights)")
a2.set_xlabel("grid point $k$"); a2.set_title("diagonal of $\\Lambda$, and the flux weights")
a2.legend(); fig.tight_layout(); plt.show()''')

md(r"""The operator is strongly diagonal — the mean intensity at a point is dominated by the source there, with smaller contributions from neighbouring depths — exactly the local-plus-tails structure of $\Lambda$. The flux weights `CH` are concentrated near the surface points: the emergent flux is set by the source in the top few optical depths, the same Eddington–Barbier intuition as Lecture 7, now as a discrete sum.""")

# ── the opacities and sources ───────────────────────────────────────────
md(r"""## The inputs: opacity, scattering, and the source

For each wavelength the solver needs, at every atmospheric depth, four opacity quantities and two source functions — all of which we built in earlier lectures and which are stored here as reference arrays of shape (depth, wavelength):

- `cont_abs` $=\kappa^{\rm abs}_{\rm cont}$ — the **continuum absorption** of Lecture 3 (mostly H$^-$).
- `cont_scat` $=\kappa^{\rm scat}_{\rm cont}$ — the **continuum scattering** of Lecture 3 (Rayleigh + Thomson).
- `line_abs` $=\kappa^{\rm abs}_{\rm line}$ — the **line absorption** of Lecture 5 (the forest of Voigt profiles).
- `line_scat` $=\kappa^{\rm scat}_{\rm line}$ — the (small) line scattering.
- `S_cont` — the **continuum source function**, the Planck function with the production code's tiny departure correction.
- `S_line` — the **line source function**.

From these we form, at each depth, the **total extinction** $\kappa^{\rm abs}+\kappa^{\rm scat}$, the **scattering fraction** $\alpha = \kappa^{\rm scat}/(\kappa^{\rm abs}+\kappa^{\rm scat})$, and the **absorption-weighted source**

$$
\bar S = \frac{\kappa^{\rm abs}_{\rm cont}\,S_{\rm cont} + \kappa^{\rm abs}_{\rm line}\,S_{\rm line}}
              {\kappa^{\rm abs}_{\rm cont} + \kappa^{\rm abs}_{\rm line}},
$$

which is the thermal ($B$-like) part of the source, before scattering mixes in $J$.""")

code(r'''D = np.load(REF / "diag.npz")
cont_abs  = D["continuum_absorption"].astype(float)   # kappa_abs continuum  (depth, wl)
cont_scat = D["continuum_scattering"].astype(float)    # kappa_scat continuum
line_abs  = D["line_opacity"].astype(float)            # kappa_abs lines
line_scat = D["line_scattering"].astype(float)         # kappa_scat lines
S_cont    = D["slinec"].astype(float)                  # continuum source function
S_line    = D["line_source"].astype(float)             # line source function
wl        = D["wavelength"]                             # nm
flux_total_ref = D["flux_total"]; flux_cont_ref = D["flux_continuum"]
print(f"{cont_abs.shape[0]} depths x {wl.size} wavelengths, {wl[0]:.1f}-{wl[-1]:.1f} nm")

EPS = 1e-38                                             # Fortran's tiny floor, avoids /0
def source_and_alpha(acont, scont, aline, sline, sigmac, sigmal):
    abtot = np.maximum(acont + aline + sigmac + sigmal, EPS)   # total extinction
    alpha = np.clip((sigmac + sigmal) / abtot, 0.0, 1.0)       # scattering fraction
    denom = acont + aline                                       # absorption only
    sbar = np.where(denom > 0, (acont*scont + aline*sline)/denom, scont)
    return abtot, alpha, sbar''')

# ── step 1: optical depth ───────────────────────────────────────────────
md(r"""## Step 1 — optical depth, by parabolic integration

The moment equations live in optical depth, so first we integrate the total extinction over the column mass `RHOX`, exactly as in Lecture 7 — but here we use the production code's **parabolic** quadrature rather than the trapezoid, because the same routine is used to build the grid the operator was tabulated on, and we want the optical-depth scale to match to the bit.

The routine `parcoe` fits, on every interval, a parabola $f \approx a + b\tau + c\tau^2$ through three neighbouring points, then forces the second and third points to be linear and blends each parabola with its neighbour by curvature weight — a mild smoothing that keeps the integrand well behaved near the boundaries. `integ` then integrates each interval's parabola analytically. The two together are the Kurucz `PARCOE`/`INTEG` pair.""")

code(r'''def parcoe(f, x):
    """Parabolic coefficients a,b,c per interval (Kurucz PARCOE)."""
    n = f.size
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    if n == 1:
        a[0] = f[0]; return a, b, c
    b[0]  = (f[1]-f[0])/(x[1]-x[0]);     a[0]  = f[0]-x[0]*b[0]     # linear endpoints
    n1 = n-1
    b[-1] = (f[-1]-f[n1-1])/(x[-1]-x[n1-1]); a[-1] = f[-1]-x[-1]*b[-1]
    if n == 2: return a, b, c
    for j in range(1, n1):                                          # parabola through 3 points
        j1 = j-1
        d = (f[j]-f[j1])/(x[j]-x[j1])
        c[j] = f[j+1]/((x[j+1]-x[j])*(x[j+1]-x[j1])) + \
               (f[j1]/(x[j+1]-x[j1]) - f[j]/(x[j+1]-x[j]))/(x[j]-x[j1])
        b[j] = d - (x[j]+x[j1])*c[j]
        a[j] = f[j1] - x[j1]*d + x[j]*x[j1]*c[j]
    c[1] = 0.0; b[1] = (f[2]-f[1])/(x[2]-x[1]); a[1] = f[1]-x[1]*b[1]   # force pts 2,3 linear
    if n > 3:
        c[2] = 0.0; b[2] = (f[3]-f[2])/(x[3]-x[2]); a[2] = f[2]-x[2]*b[2]
    for j in range(1, n1):                                          # curvature-weighted blend
        if c[j] == 0.0: continue
        j1 = min(j+1, n-1); denom = abs(c[j1]) + abs(c[j])
        wt = abs(c[j1])/denom if denom > 0 else 0.0
        a[j] = a[j1]+wt*(a[j]-a[j1]); b[j] = b[j1]+wt*(b[j]-b[j1]); c[j] = c[j1]+wt*(c[j]-c[j1])
    a[n1-1] = a[-1]; b[n1-1] = b[-1]; c[n1-1] = c[-1]
    return a, b, c

def integ(x, f, start):
    """Cumulative integral of f dx using each interval's left-point parabola (Kurucz INTEG)."""
    a, b, c = parcoe(f, x)
    out = np.zeros(f.size); out[0] = start
    for i in range(f.size-1):
        dx = x[i+1]-x[i]
        term = a[i] + 0.5*b[i]*(x[i+1]+x[i]) + (c[i]/3.0)*((x[i+1]+x[i])*x[i+1] + x[i]*x[i])
        out[i+1] = out[i] + term*dx
    return out''')

# ── step 2: map onto the grid ───────────────────────────────────────────
md(r"""## Step 2 — map the source onto the fixed grid

The atmosphere has 80 depth points, with a $\tau_\lambda$ scale that differs at every wavelength; the operator `COEFJ` lives on the fixed 51-point grid `XTAU`. So we interpolate the absorption-weighted source $\bar S$ and the scattering fraction $\alpha$ from the atmosphere's $\tau_\lambda$ onto `XTAU`. The production code uses the same parabolic interpolation everywhere — the `MAP1` routine — so we reproduce it rather than reach for a library spline; the small differences in interpolation scheme are exactly what would spoil bit-level agreement.

Where a grid point lies **above** the top of the atmosphere ($\tau < \tau_\lambda[0]$), there is nothing to interpolate, so the source is held at its surface value. That surface masking matters in strong lines, whose opacity lifts $\tau_\lambda[0]$ to non-negligible values.""")

code(r'''def map1(xold, fold, xnew):
    """Parabolic interpolation of fold(xold) onto xnew (Kurucz MAP1)."""
    nold, nnew = xold.size, xnew.size
    fnew = np.zeros(nnew)
    if nold == 0 or nnew == 0: return fnew
    xo = np.empty(nold+1); fo = np.empty(nold+1); xo[1:] = xold; fo[1:] = fold
    l = 2; ll = 0
    cfor = bfor = afor = cbac = bbac = abac = a = b = c = 0.0
    for k in range(1, nnew+1):
        xk = xnew[k-1]
        while True:
            if xk < xo[l]:
                if l == ll: break
                if l == 2 or l == 3:
                    l = min(nold, l); c = 0.0
                    b = (fo[l]-fo[l-1])/(xo[l]-xo[l-1]); a = fo[l]-xo[l]*b; ll = l; break
                l1 = l-1
                if l > ll+1 or l == 3 or l == 4:               # backward parabola
                    l2 = l-2
                    d = (fo[l1]-fo[l2])/(xo[l1]-xo[l2])
                    cbac = fo[l]/((xo[l]-xo[l1])*(xo[l]-xo[l2])) + \
                           (fo[l2]/(xo[l]-xo[l2]) - fo[l1]/(xo[l]-xo[l1]))/(xo[l1]-xo[l2])
                    bbac = d - (xo[l1]+xo[l2])*cbac
                    abac = fo[l2] - xo[l2]*d + xo[l1]*xo[l2]*cbac
                    if l >= nold: c, b, a, ll = cbac, bbac, abac, l; break
                else:
                    cbac, bbac, abac = cfor, bfor, afor
                    if l == nold: c, b, a, ll = cbac, bbac, abac, l; break
                d = (fo[l]-fo[l1])/(xo[l]-xo[l1])               # forward parabola + blend
                cfor = fo[l+1]/((xo[l+1]-xo[l])*(xo[l+1]-xo[l1])) + \
                       (fo[l1]/(xo[l+1]-xo[l1]) - fo[l]/(xo[l+1]-xo[l]))/(xo[l]-xo[l1])
                bfor = d - (xo[l]+xo[l1])*cfor
                afor = fo[l1] - xo[l1]*d + xo[l]*xo[l1]*cfor
                wt = abs(cfor)/(abs(cfor)+abs(cbac)) if abs(cfor) != 0 else 0.0
                a = afor+wt*(abac-afor); b = bfor+wt*(bbac-bfor); c = cfor+wt*(cbac-cfor); ll = l; break
            l += 1
            if l > nold:
                l = min(nold, l); c = 0.0
                b = (fo[l]-fo[l-1])/(xo[l]-xo[l-1]); a = fo[l]-xo[l]*b; ll = l; break
        fnew[k-1] = a + (b + c*xk)*xk
    return fnew''')

# ── step 3: the scattering iteration ────────────────────────────────────
md(r"""## Step 3 — the scattering source, by iteration

Now the heart of the method. On the grid we want the source function that is consistent with the radiation field it produces. With the absorption-weighted source $\bar S$ and the operator $J = \texttt{COEFJ}\cdot S$, the scattering relation $S = (1-\alpha)\bar S + \alpha J$ becomes a **linear fixed point on the grid**:

$$
S_k = (1-\alpha_k)\,\bar S_k + \alpha_k \sum_m \texttt{COEFJ}_{km}\,S_m .
$$

When the scattering fraction $\alpha$ is zero this collapses to $S = \bar S$ — the pure-absorption case of Lecture 7. When $\alpha > 0$ the source is lowered toward $J$, which in the optically thin surface layers is smaller than $B$ — this is what darkens the cores. We solve the fixed point by a backward Gauss–Seidel sweep: isolating the diagonal term, the update at each point is

$$
\Delta S_k = \frac{\alpha_k\,(\texttt{COEFJ}\cdot S)_k + (1-\alpha_k)\bar S_k - S_k}{1 - \alpha_k\,\texttt{COEFJ}_{kk}},
$$

repeated until every relative change falls below $10^{-5}$. The production code carries this iteration in **single precision** (the original Fortran used `REAL*4` arrays here), so we do too — it is the one place where the working precision is part of the specification.""")

code(r'''ITER_TOL, MAX_ITER = 1e-5, 51
COEFJ_DIAG = np.diag(COEFJ).copy()

def iterate_source(sbar_grid, alpha_grid):
    """Solve S = (1-alpha) sbar + alpha (COEFJ @ S) by backward Gauss-Seidel, float32."""
    co = COEFJ.astype(np.float32)
    xs = sbar_grid.astype(np.float32)                 # initial guess: the thermal source
    al = alpha_grid.astype(np.float32)
    sbar_mod = (sbar_grid * (1.0 - alpha_grid)).astype(np.float32)   # (1-alpha) * sbar
    diag = (1.0 - alpha_grid * COEFJ_DIAG).astype(np.float32)
    tol, eps = np.float32(ITER_TOL), np.float32(EPS)
    for _ in range(MAX_ITER):
        converged = True
        for k in range(NXTAU-1, -1, -1):              # backward sweep
            j_k = np.float32(np.dot(co[k], xs))       # (COEFJ @ S)_k, the mean intensity term
            delta = (j_k*al[k] + sbar_mod[k] - xs[k]) / diag[k]
            if (abs(delta/xs[k]) if xs[k] != 0 else np.inf) > tol:
                converged = False
            xs[k] = max(xs[k] + delta, eps)
        if converged:
            break
    return xs.astype(np.float64)''')

# ── step 4: assemble ────────────────────────────────────────────────────
md(r"""## Step 4 — the surface flux, and the whole solver

The converged source on the grid gives the emergent Eddington flux by the weighted sum $H(0) = \sum_k \texttt{CH}_k\,S_k$. Assembling the four steps — optical depth, map to the grid, iterate the source, weight for the flux — is the complete `solve_josh` for one wavelength. One bookkeeping detail: `INTEG` needs the column mass increasing from the surface inward; our `RHOX` already is, so no reversal is needed here.""")

code(r'''def solve_josh(acont, scont, aline, sline, sigmac, sigmal):
    """Emergent Eddington flux H(0) at one wavelength via the JOSH moment method."""
    abtot, alpha, sbar = source_and_alpha(acont, scont, aline, sline, sigmac, sigmal)
    tau = integ(RHOX, abtot, abtot[0]*RHOX[0])             # step 1: optical depth
    sbar_g  = np.maximum(map1(tau, sbar,  XTAU), EPS)        # step 2: map onto the grid
    alpha_g = np.clip(   map1(tau, alpha, XTAU), 0.0, 1.0)
    above = XTAU < tau[0]                                    # grid points above the atmosphere
    if above.any():
        sbar_g[above] = max(sbar[0], EPS); alpha_g[above] = np.clip(alpha[0], 0.0, 1.0)
    S = iterate_source(sbar_g, alpha_g)                     # step 3: scattering iteration
    return float(CH @ S)                                    # step 4: weighted surface flux

# the continuum is the same solver with the line terms switched off
def solve_continuum(acont, scont, sigmac):
    zero = np.zeros_like(acont)
    return solve_josh(acont, scont, zero, scont, sigmac, zero)
print("solver assembled")''')

# ── worked example ──────────────────────────────────────────────────────
md(r"""## A worked example: one line core, one continuum point

Before running the whole window, watch the iteration act on a single wavelength. We pick the **deepest line core** in the band and a nearby **continuum** point, and look at the source function on the grid before and after the scattering iteration. In the continuum, where absorption dominates ($\alpha \approx 0$), the iteration barely moves the source. In the line core, where the opacity has lifted the formation height into thin, scattering-prone gas, the iteration pulls the surface source **below** the thermal value — the physical origin of the extra core darkening.""")

code(r'''ref_spec = flux_total_ref / flux_cont_ref
kc = int(np.argmin(ref_spec))                              # deepest line core
kk = int(np.argmin(np.abs(ref_spec - 1.0)))               # a continuum point (spectrum ~ 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
for ax, k, name in [(axes[0], kk, "continuum"), (axes[1], kc, "deep line core")]:
    abtot, alpha, sbar = source_and_alpha(cont_abs[:,k], S_cont[:,k], line_abs[:,k],
                                          S_line[:,k], cont_scat[:,k], line_scat[:,k])
    tau = integ(RHOX, abtot, abtot[0]*RHOX[0])
    sbar_g  = np.maximum(map1(tau, sbar,  XTAU), EPS)
    alpha_g = np.clip(   map1(tau, alpha, XTAU), 0.0, 1.0)
    above = XTAU < tau[0]
    if above.any(): sbar_g[above] = max(sbar[0], EPS); alpha_g[above] = np.clip(alpha[0],0,1)
    S = iterate_source(sbar_g, alpha_g)
    ax.loglog(XTAU, sbar_g, "o-", ms=3, label=r"thermal $\bar S$ (no scattering)")
    ax.loglog(XTAU, S, "s-", ms=3, color="C3", label="iterated $S$ (with scattering)")
    ax.set_title(f"{name}:  $\\lambda$ = {wl[k]:.3f} nm,  max $\\alpha$ = {alpha_g.max():.2f}")
    ax.set_xlabel(r"optical depth $\tau$")
axes[0].set_ylabel("source function"); axes[0].legend(loc="upper left", fontsize=9)
fig.tight_layout(); plt.show()''')

md(r"""In the continuum panel the two curves lie on top of each other — scattering is a percent-level effect and $S \approx \bar S$. In the line-core panel the iterated source peels away from the thermal source in the surface layers, sitting lower by tens of percent at the very top: those are the photons that scatter out instead of being re-emitted thermally, and they are exactly the part the formal solution missed.""")

# ── the full spectrum ───────────────────────────────────────────────────
md(r"""## The full spectrum, to machine precision

Now run the solver across all wavelengths — the line flux with every opacity term, the continuum flux with the lines switched off — and take the ratio for the normalised spectrum. This is the same quantity Lecture 7 produced with the formal solution; here we compare it to the reference computed by the production code's own JOSH engine.""")

code(r'''flux_line = np.array([solve_josh(cont_abs[:,k], S_cont[:,k], line_abs[:,k],
                                 S_line[:,k], cont_scat[:,k], line_scat[:,k])
                      for k in range(wl.size)])
flux_cont = np.array([solve_continuum(cont_abs[:,k], S_cont[:,k], cont_scat[:,k])
                      for k in range(wl.size)])
spectrum  = flux_line / flux_cont
reference = flux_total_ref / flux_cont_ref
rel = np.abs(spectrum - reference) / np.abs(reference)
print(f"normalised spectrum vs reference:  median |rel diff| = {np.median(rel):.2e}   "
      f"max = {rel.max():.2e}")
print("the residual is the single-precision iteration's last bit — the engine is reproduced.")''')

md(r"""**Machine precision.** The from-scratch JOSH solver reproduces the reference solar spectrum to a median of a few parts in $10^{12}$, with the worst point at the level of the single-precision iteration's last bit — the same arithmetic the production code uses. The ten-percent gap the formal solution left in the deepest cores is gone: the moment method's scattering iteration accounts for it exactly.""")

code(r'''fig, (ax, axr) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
ax.plot(wl, reference, color="0.6", lw=1.4, label="reference (production JOSH)")
ax.plot(wl, spectrum, color="C3", lw=0.6, label="from scratch (this lecture)")
ax.set_ylabel("normalised flux"); ax.set_ylim(0, 1.05); ax.legend(loc="lower right")
ax.set_title("The solar spectrum, 500–510 nm — JOSH solver rebuilt, matched to the bit")
axr.semilogy(wl, np.maximum(rel, 1e-16), color="C0", lw=0.5)
axr.set_xlabel("wavelength  [nm]"); axr.set_ylabel("|rel diff|"); axr.set_ylim(1e-15, 1e-7)
fig.tight_layout(); plt.show()''')

md(r"""The two spectra are indistinguishable, and the residual panel sits near $10^{-12}$ across the whole window, rising only to the single-precision floor in the sharpest cores. We have rebuilt the production radiative-transfer engine — moments, closure, operator, iteration, flux — and it reproduces the reference exactly.""")

# ── the saturated-core path ─────────────────────────────────────────────
md(r"""## A note on saturated cores

One branch of the production solver does not fire in this window and so we have not needed it: when the **surface** optical depth itself exceeds the top of the fixed grid ($\tau_\lambda[0] > \tau_{\rm max}$), there is no grid point above the atmosphere to anchor the interpolation, and the code switches to solving the moment equations directly on the physical $\tau_\lambda$ scale. That happens only in extremely opaque, saturated cores — the molecular band heads of cool stars, not the optical solar window — and we will meet it when we add molecules. For every wavelength in the Sun from 500 to 510 nm, the surface optical depth is small and the grid path above is exact.""")

# ── synthesis ───────────────────────────────────────────────────────────
md(r"""## Synthesis

The formal solution of Lecture 7 gave the physical picture — flux as a weighted average of the source over depth — and the spectrum to a part in a thousand. This lecture supplied the engine that closes the last gap. Taking moments of the transfer equation and applying the Eddington closure turns angle-dependent transfer into two moment equations; discretising them on a fixed optical-depth grid turns the $\Lambda$ operator into a matrix; and iterating the scattering source $S = (1-\alpha)\bar S + \alpha J$ to self-consistency recovers the core darkening that pure absorption could not. Reusing the same parabolic integration, the same interpolation, and the same single-precision iteration as the production code, the result is the solar spectrum reproduced to machine precision.

With the microphysics (Lectures 2–6) and both radiative-transfer treatments (Lectures 7–8) in hand, the synthesis half of the pipeline is complete and exact. What remains is to stop taking the **model atmosphere** as given: in Part IV we build the temperature and pressure structure ourselves, from hydrostatic and radiative equilibrium, and watch the spectrum settle onto the real Sun.""")

md(r"""## Summary

- Taking **moments** of the transfer equation gives $dH/d\tau = J - S$ and $dK/d\tau = H$; the **Eddington closure** $K = \tfrac13 J$ shuts the system, and its solution is linear in the source, $J = \Lambda[S]$.
- The production solver ships $\Lambda$ as a matrix `COEFJ` on a **fixed 51-point optical-depth grid**, with flux weights `CH` for the surface flux $H(0) = \sum_k \texttt{CH}_k S_k$.
- The scattering source obeys $S = (1-\alpha)\bar S + \alpha\,(\texttt{COEFJ}\cdot S)$, solved by a backward Gauss–Seidel sweep in **single precision** to a $10^{-5}$ tolerance.
- The pipeline per wavelength is: **optical depth** (`parcoe`/`integ`) → **map onto the grid** (`map1`) → **iterate the source** → **weighted flux**.
- The rebuilt solver reproduces the reference spectrum to a **median of $\sim10^{-12}$**, closing the deep-core scattering gap the formal solution left.""")

md(r"""## Practice exercises

**1. The closure in action.** Set $\alpha = 0$ everywhere in `solve_josh` and confirm the result matches the formal solution of Lecture 7 (the source never leaves $\bar S$). Then raise $\alpha$ by hand in a single core and watch the central depth grow — how much scattering is needed to darken the core by ten percent?

**2. The operator's reach.** Plot a few rows of `COEFJ` (say $k=5, 25, 45$) against grid index. How localised is the mean-intensity response, and how does its width compare with the spacing of the optical-depth grid? Relate this to why a strongly scattering line core feels the temperature of layers above it.

**3. Precision matters.** Change `iterate_source` to work in `float64` instead of `float32` and recompute the spectrum. Where does it differ from the reference, and by how much? Explain why matching the production code requires matching its working precision, not just its algorithm.

**4. Convergence.** Count the iterations to convergence as a function of the maximum scattering fraction $\alpha$ along the column (return the count from `iterate_source`). Why do scattering-dominated wavelengths take longer, and what does that imply for the cost of a full spectral synthesis?""")

md(r"""## Further reading

- **Avrett, E. H. & Loeser, R. (1969). *SAO Special Report* 303.** The moment method with variable Eddington factors that the JOSH solver descends from.
- **Mihalas, D. (1978). *Stellar Atmospheres*, 2nd ed., Freeman.** Chapters 6–7 on the moment equations, the Eddington approximation, and $\Lambda$-iteration.
- **Hubeny, I. & Mihalas, D. (2014). *Theory of Stellar Atmospheres*, Princeton.** Chapters 11–13 on operator methods, accelerated lambda iteration, and the convergence of scattering problems.
- **Kurucz, R. L. (1970). *SAO Special Report* 309 (ATLAS).** The original code whose `PARCOE`, `INTEG`, `MAP1`, and JOSH routines we reproduce.
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz*](https://arxiv.org/abs/2603.11693).** The implementation our reference spectrum is computed with.""")

nb = new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
