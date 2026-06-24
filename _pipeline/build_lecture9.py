#!/usr/bin/env python
"""Lecture 9 — Hydrostatic Equilibrium & Temperature Structure: build the grey model
atmosphere itself (which Lectures 1-8 took as GIVEN) from (Teff, logg) to machine
precision.  Reproduces pykurucz's grey cold start (generate_grey_atm -> the ATLAS12
TTAUP hydrostatic integrator) and matches reference/L1.npz grey_T, grey_pgas, grey_rhox
BIT-EXACT.  Self-contained: imports only numpy / matplotlib / pathlib; benchmarks against
the shipped reference/L1.npz data file.  No pykurucz import.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture9.ipynb"
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

# ── title ────────────────────────────────────────────────────────────────
md(r"""# Lecture 9 — Hydrostatic Equilibrium & Temperature Structure

*Stellar Spectroscopy from Scratch — rebuilding the physics of ATLAS and SYNTHE from first principles*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*Every result in this book is checked against reference values computed with [**pykurucz**](https://arxiv.org/abs/2603.11693) — a pure-Python implementation of Kurucz's ATLAS12 and SYNTHE — shipped beside the lectures as small data files, so the notebooks need only NumPy to run.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Explain why the first lecture's closed-form $P_{\rm total}=g\tau$ is the exact **zero-boundary, $\kappa\equiv1$ total-pressure limit** of the hydrostatic equation, and why reproducing the production code's *stored gas pressure* to the last digit needs its boundary convention, its radiation-pressure treatment, and its discrete integrator.
- Set up the **radiation-pressure correction** $P_{\rm gas}=P_{\rm total}-P_{\rm rad}$ and explain why the structure code carries it even where it is negligible.
- Integrate hydrostatic equilibrium in **log pressure** and write the chain rule cleanly, $\Delta p = (g/\kappa)(\tau/P_{\rm total})\,\Delta\ln\tau$, so the math matches the code.
- Implement the production code's **predictor-corrector** multistep integrator — including its boundary seed and its evaluate-then-check convergence ordering — and reproduce the reference grey atmosphere's gas pressure and column mass to **machine precision**, closing the residual the first lecture deferred.""")

# ── introduction ───────────────────────────────────────────────────────────
md(r"""## Introduction: building the atmosphere we have been given

For eight lectures the **model atmosphere** has been a given. We took its run of temperature, gas pressure, and density with depth — the columns of a `.atm` file — and on top of it built the equation of state, the continuous and line opacities, and the radiative-transfer solver, until we reproduced the solar spectrum to machine precision. But where did that atmosphere come from? In this static, plane-parallel setup the two *structural* equilibrium constraints are **hydrostatic equilibrium**, that the gas neither collapses under its own weight nor blows away, and **radiative equilibrium**, that the energy carried outward by radiation is conserved at every depth; closing the model also requires an equation of state, opacities, a composition, boundary conditions, and a convection treatment. This lecture builds the first structural constraint. It is the start of the **inverse half** of the course: instead of taking the structure and computing the spectrum, we compute the structure itself from the two numbers that fix the grey cold start of a star — its effective temperature $T_{\rm eff}$ and its surface gravity $\log g$. (A full model also fixes the composition and microturbulence; in the simplified grey setup here, $T_{\rm eff}$ and $\log g$ are enough.)

Lecture 1 already built most of the pieces we need here, and we will lean on them rather than re-derive them. It wrote down the **grey/Hopf temperature law** $T(\tau)$, laid out the **80-layer optical-depth grid**, and introduced **hydrostatic equilibrium** in optical depth, $dP/d\tau = g/\kappa$. It also took one shortcut: on the cold start it set $\kappa\equiv1$, integrated the balance by hand to the one-line result $P_{\rm total}=g\tau$, and found it agreed with the reference only to about one part in $10^{5}$.

That one part in $10^{5}$ is worth being precise about, because it is *not* a flaw in the closed form. For $\kappa\equiv1$ and a zero-pressure top boundary, $P_{\rm total}=g\tau$ is the **exact analytic solution** of the hydrostatic equation, not an approximation to it. The residual comes from the things the one-liner left out: the production code's boundary seed, its finite-grid predictor-corrector, its radiation-pressure increment and subtraction, and the fact that the reference array is a *gas* pressure while $g\tau$ is a *total* pressure. **Reproducing that stored gas pressure to the last bit is the one genuinely new thing this lecture does.** We recap the temperature and the grid quickly, then spend our effort where Lecture 1 stopped: on the same boundary convention, the same radiation-pressure treatment, and the exact predictor-corrector that integrates the equation in log pressure. With those in hand the last digits fall into place — gas pressure and column mass become bit-for-bit identical to the reference grey model that ATLAS12 starts its own iteration from.

![From $T_{\rm eff}$ and $\log g$: the grey temperature law and the hydrostatic integration of $dP/d\tau=g/\kappa$ down the optical-depth grid give the run of temperature, pressure, and density with depth.](resources/figures/s8_hydrostatic.png)""")

# ── the imports + reference ────────────────────────────────────────────────
md(r"""## Setup and the reference

We import only NumPy and Matplotlib. The benchmark target is the same `reference/L1.npz` the first lecture used: it carries the grey optical-depth grid `grey_tau`, the grey temperature `grey_T`, the gas pressure `grey_pgas`, and the column mass `grey_rhox`, all computed once by the production code on the solar parameters. The first lecture matched `grey_tau` and `grey_T` exactly and `grey_pgas`/`grey_rhox` to $\sim2\times10^{-5}$; by the end of this lecture all four match to the bit.""")

code(r'''import pathlib
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5})

# the benchmark target: the grey structure the production code computed for the Sun
REF = np.load(pathlib.Path("..") / "reference" / "L1.npz")

# the Sun's two defining numbers: effective temperature and log surface gravity
TEFF, LOGG = 5770.0, 4.44                  # [K], [cgs]
g_cgs = 10.0 ** LOGG                       # surface gravity g [cm s^-2]

print(f"target: grey atmosphere for Teff = {TEFF:.0f} K, log g = {LOGG}  ->  g = {g_cgs:.4g} cm/s^2")
print(f"reference arrays: {[k for k in REF.files]}")''')

# ── hydrostatic equilibrium ─────────────────────────────────────────────────
md(r"""## Hydrostatic equilibrium, in the form the codes integrate (Lecture 1)

The first lecture introduced hydrostatic equilibrium — the balance between the pressure gradient and the weight of the overlying gas — and rewrote it in **optical depth** $\tau$, the variable the radiation field cares about. With height $z$ measured upward, $\tau$ and column mass increasing inward, gravity $g$ constant across the thin photosphere, and $\kappa$ a Rosseland mass opacity in $\mathrm{cm^2\,g^{-1}}$, the plane-parallel hydrostatic equation becomes

$$
\frac{dP}{d\tau} = \frac{g}{\kappa}.
$$

We take that as our starting point. Read it as a statement about the column mass: since $dP/d m = g$, integrating it gives the weight per unit area of the overlying gas, $g\,m$. With the usual zero-pressure top boundary this is just $P_{\rm total} = g\,m$ — up to the chosen top-boundary constant, which is exactly the surface radiation pressure $P_{\rm rad}(0)$ when the boundary carries no gas, and which we will see the integrator handle explicitly later. Here $m$ is the **column mass**, the grams of material above a square centimetre; Kurucz calls this quantity `RHOX`, and it is a single compound symbol $m=\int_z^\infty\rho\,dz'$, *not* the local density $\rho$ multiplied by a depth coordinate $x$. The optical-depth form makes the opacity explicit: where the gas is more opaque (large $\kappa$), a given range of $\tau$ spans less pressure, because you reach optical depth one in less material. In this atmosphere-construction context, $\tau$ means **Rosseland** optical depth and $\kappa$ the **Rosseland-mean opacity**, unless stated otherwise.

So far this is all Lecture 1. What that lecture did *not* do is integrate the equation on the production grid: it set $\kappa\equiv1$ (i.e. $\kappa_{\rm Ross}=1\,\mathrm{cm^2\,g^{-1}}$ in the code's cgs convention) and read off the analytic $P_{\rm total}=g\tau$ in one line. The rest of this lecture reproduces the discrete integration the reference uses. To do it we need two inputs at every depth — the temperature (which sets $P_{\rm rad}$ and, through the equation of state, the density) and the opacity $\kappa(\tau)$ — so we recap those two first, then build the integrator.""")

# ── the temperature law ─────────────────────────────────────────────────────
md(r"""## Recap (Lecture 1): the grey/Hopf temperature

The temperature structure is the *input* to the pressure integration. Lecture 1 derived it in full from the grey transfer equation; here we only need the result, so this is a one-paragraph recap. The grey/Hopf law is $T(\tau) = T_{\rm eff}\,[\tfrac34(0.710 + \tau - 0.1331\,e^{-3.4488\tau})]^{1/4}$ — Kurucz's analytic (exponential) fit to the Hopf function $q(\tau)$, with $q$ rising from $1/\sqrt3\approx0.577$ at the surface to $0.710$ deep down (see Lecture 1 for the derivation and the usual $\tau\sim2/3$ photospheric scale). The code below is the identical fit from Lecture 1: it evaluates the bracket $\tfrac34(\tau+q)$, floors it at $10^{-300}$ so the fourth root can never see a negative argument (a safety guard that never triggers for a real atmosphere), and takes the $1/4$ power. We re-include it here because it feeds the pressure integration that follows.""")

code(r'''def grey_temperature(teff, tau):
    """Eddington-Kurucz grey T(tau): Kurucz's analytic fit to the Hopf function q(tau)."""
    # the bracket (3/4)(tau + q(tau)), with q(tau) the exponential Hopf fit
    bracket = 0.75 * (0.710 + tau - 0.1331 * np.exp(-3.4488 * tau))

    # T = Teff * bracket^(1/4); the floor keeps the fourth root off a negative argument
    return float(teff) * np.power(np.maximum(bracket, 1e-300), 0.25)''')

# ── the optical-depth grid ──────────────────────────────────────────────────
md(r"""## Recap (Lecture 1): the optical-depth grid

The same grid the first lecture used. It is the literal ATLAS12 `CALCULATE` card defaults — **80 layers** uniform in $\log\tau$, the first at $\log\tau=-6.875$ and spaced **$0.125$ dex** apart, spanning nearly ten decades from $\tau\simeq1.3\times10^{-7}$ at the top ($\log\tau=-6.875$) to $\tau=10^{3}$ at the base ($\log\tau=-6.875+79\times0.125=3.0$), well below the photosphere (Lecture 1 explains why this range and spacing). The one fact we will use heavily below: because the grid is uniform in $\log\tau$, the log-tau step is a *constant*, and it becomes the natural increment of the integration.

The code builds the grid by raising 10 to the linear ramp $\log\tau_j = -6.875 + 0.125\,j$ for $j=0\dots79$, then evaluates the recapped temperature law on it. The print line spot-checks the endpoints and the photosphere.""")

code(r'''NRHOX, TAU1LG, STEPLG = 80, -6.875, 0.125      # ATLAS12 CALCULATE card defaults
j = np.arange(NRHOX, dtype=np.float64)           # layer index 0..79

# the grid: 10 raised to the linear log-tau ramp, 80 layers 0.125 dex apart, surface -> deep
tau = np.power(10.0, TAU1LG + j * STEPLG)

# the recapped grey/Hopf temperature law, evaluated on that grid
T = grey_temperature(TEFF, tau)

print(f"{NRHOX} layers,  tau = {tau[0]:.3e} .. {tau[-1]:.3e}")
print(f"T(top) = {T[0]:.1f} K    T(tau~1) ~ {grey_temperature(TEFF, 1.0):.1f} K    T(bottom) = {T[-1]:.1f} K")''')

md(r"""The grid and the temperature are the first two reference arrays, and they match to the bit straight away — they are the same Hopf fit on the same grid the first lecture built. We confirm that with a small helper, `check`, that we will reuse for the full benchmark: it computes the worst relative difference against the reference array and flags `bit-exact` when the two are equal to the last bit.""")

code(r'''def check(name, got, ref):
    """Report max|relative diff| between a from-scratch array and the reference; flag bit-exact."""
    got, ref = np.asarray(got, float), np.asarray(ref, float)

    # mask out exact zeros, where a relative difference is undefined
    m = np.abs(ref) > 0
    rel = np.zeros_like(ref)
    rel[m] = np.abs(got[m] - ref[m]) / np.abs(ref[m])

    tag = "bit-exact" if np.array_equal(got, ref) else f"max|rel| = {rel.max():.2e}"
    print(f"  {name:12s}  {tag}")
    return rel.max()

# the grid and the temperature are the same Lecture-1 objects, so both come back bit-exact
check("grey_tau", tau, REF["grey_tau"])        # same grid
check("grey_T",   T,   REF["grey_T"])          # same Hopf fit''')

# ── radiation pressure ──────────────────────────────────────────────────────
md(r"""## Radiation pressure

This is new content — Lecture 1 mentioned $P_{\rm rad}$ in passing but the integrator here actually carries it, so it is worth getting the definition exactly right. The total pressure has two parts, gas and radiation: $P_{\rm total} = P_{\rm gas} + P_{\rm rad}$. The hydrostatic balance supports the **total**, but the equation of state and the line opacities care about the **gas** pressure, so we will subtract $P_{\rm rad}$ at the end.

Be careful with the coefficient. The radiation **energy density** of an isotropic field is $a_{\rm rad}T^4$ with $a_{\rm rad}\approx7.566\times10^{-15}\,\mathrm{erg\,cm^{-3}\,K^{-4}}$, while the corresponding radiation **pressure** is one third of that, $P_{\rm rad}=\tfrac13 a_{\rm rad}T^4 = \tfrac{4\sigma}{3c}T^4$.

This $\tfrac13 a_{\rm rad}T^4$ form is the local, diffusion/Eddington-style radiation-pressure correction the code uses: it is exact for an isotropic blackbody field and a good approximation in the deep, optically thick layers, but near the surface the radiation field is forward-peaked and anisotropic, so the true angular radiation pressure need not equal $\tfrac13 a_{\rm rad}T^4$ there. ATLAS adopts it as the standard structural approximation throughout. Kurucz carries the pressure coefficient $\tfrac13 a_{\rm rad}$ as the constant $2.521\times10^{-15}$. The code also floors the temperature inside the fourth power at $T_{\rm eff}^4/2$, so $P_{\rm rad}$ never drops below a fixed fraction of its photospheric value in the cool upper layers — a stabiliser that keeps the cold start well behaved:

$$
P_{\rm rad}(\tau) = 2.521\times10^{-15}\,\max\!\big(T^4,\ \tfrac12 T_{\rm eff}^4\big).
$$

For this solar cold start the radiation-pressure *correction to the hydrostatic run* is small, but it is retained because the reference code retains it and because it matters in hotter, lower-gravity atmospheres. What enters the integration is the **run** of $P_{\rm rad}$ relative to the surface, $P_{\rm rad}(\tau)-P_{\rm rad}(0)$, since only the *gradient* of pressure matters. The code therefore keeps two arrays: `pradk` is the absolute radiation pressure at each layer, while `prad = pradk - pradk[0]` is the depth-dependent increment relative to the top layer (so `prad[0]` is exactly zero by construction). The hydrostatic integration only ever needs this increment; the surface constant is absorbed into the boundary pressure. The turbulent pressure `pturb` is zero on the cold start.""")

code(r'''# absolute radiation pressure at each layer, with Kurucz's T^4 floor at Teff^4/2
pradk = 2.521e-15 * np.maximum(T ** 4, TEFF ** 4 / 2.0)

# the depth-dependent run relative to the top layer, which is what the integration needs
prad = pradk - pradk[0]                       # prad[0] = 0 by construction

# turbulent pressure: identically zero on the cold start
pturb = np.zeros(NRHOX)

print(f"P_rad(top) = {pradk[0]:.3e}   P_rad(bottom) = {pradk[-1]:.3e} dyn/cm^2")
print("  (only the increment P_rad(tau) - P_rad(0) enters the hydrostatic balance)")''')

# ── the cold-start opacity ──────────────────────────────────────────────────
md(r"""## Why the cold start uses $\kappa\equiv1$

Hydrostatic equilibrium $dP/d\tau = g/\kappa$ needs the opacity $\kappa(\tau)$ — but the opacity depends on the pressure and temperature through the equation of state and the photo-absorption cross-sections, which we cannot evaluate until we *have* the structure. This is the bootstrap problem at the heart of building an atmosphere: structure needs opacity, opacity needs structure.

The production code breaks the loop with a **cold start**. On the very first pass there is no opacity table yet, so the Rosseland-mean opacity is set to a placeholder of **unity at every layer**, $\kappa_{\rm Ross}\equiv1$. (Internally the opacity table — `ROSSTAB` — is empty, and an empty table returns $1.0$.) This is deliberately crude: it gives a first guess of the pressure structure that is in the right ballpark, which ATLAS then refines over many iterations, rebuilding the real Rosseland-mean opacity from the equation of state and feeding it back into the hydrostatic integration until the structure stops changing. We are reproducing that **first guess** — the grey model the iteration starts from — so $\kappa\equiv1$ throughout. There is one cosmetic exception: the top layer is seeded with $\kappa=0.1$ purely to set the boundary value of the integration, as we see next.""")

# ── log pressure ────────────────────────────────────────────────────────────
md(r"""## Integrating in log pressure, and why

Now the new work begins: the actual integration of $dP/d\tau = g/\kappa$ down the grid. The natural move would be to step in $P$ directly, but two facts make **log pressure** the better variable. First, the pressure spans eight or nine decades from the top of the atmosphere to the bottom, so a fixed step in $P$ would be hopelessly coarse at the top and wastefully fine at the bottom, while a step in $\log P$ resolves every decade equally. Second, the grid is uniform in $\log\tau$, and on the cold start with $\kappa$ constant the solution $P=g\tau$ is a power law, so $\log P$ is *linear* in $\log\tau$ — the integrand is smoothest in log-log variables, where a low-order multistep formula is most accurate.

The chain rule is cleanest in natural logs. Let $p\equiv\ln P_{\rm total}$ and $u\equiv\ln\tau$. Then, applying $d/du$ through $p$, $P_{\rm total}$, and $\tau$ in turn,

$$
\frac{dp}{du}
= \frac{1}{P_{\rm total}}\,\frac{dP_{\rm total}}{d\tau}\,\frac{d\tau}{du}
= \frac{1}{P_{\rm total}}\,\frac{g}{\kappa}\,\tau
= \frac{g}{\kappa}\,\frac{\tau}{P_{\rm total}},
$$

using $dP_{\rm total}/d\tau = g/\kappa$ from hydrostatic equilibrium and $d\tau/du = \tau$ (since $u=\ln\tau$). Multiplying by the constant step $\Delta u = \Delta\ln\tau$ gives the quantity the code carries from layer to layer,

$$
\Delta p_j \;=\; \frac{g}{\kappa_j}\,\frac{\tau_j}{P_{{\rm total},j}}\,\Delta\ln\tau .
$$

This $\Delta p_j$ is *not* the exact finite difference $p_{j+1}-p_j$; it is the continuous derivative $dp/du$ **scaled by the grid step** — the standard $h\cdot f(y_n)$ term of a multistep ODE solver, which the predictor and corrector then blend across several layers to extrapolate the actual step. The step $\Delta\ln\tau = \ln(\tau_{j+1}/\tau_j)$ is the same constant for every interval because the grid is uniform in $\log\tau$ (its value is $\ln 10 \times 0.125$). We will compute this scaled derivative — call it `dplog` — at each layer and use it both to **predict** the next layer's pressure and to **correct** it. The cell below just forms that constant step and confirms it equals $\ln 10 \times 0.125$.""")

code(r'''# the natural-log tau step, a constant across the uniform-in-log-tau grid
dlg_tau = np.log(tau[1] / tau[0])
print(f"d(ln tau) per layer = {dlg_tau:.6f}   (= ln(10) * 0.125 = {np.log(10)*0.125:.6f})")''')

# ── predictor-corrector ─────────────────────────────────────────────────────
md(r"""## The predictor-corrector integrator

This is the heart of the lecture — the exact integrator that replaces Lecture 1's one-line $P=g\tau$. We integrate layer by layer from the top down, carrying a short **history** of the last few values of $p=\ln P$ and of the derivative $\Delta p$. Each layer is done in two stages, predict then correct.

**Predictor.** Extrapolate $p_j$ from the history. The very first layer ($j=0$) has no history, so its pressure is set directly from the seeded boundary opacity, $p_0 = \ln(g\,\tau_0/\kappa_0)$ with $\kappa_0=0.1$. The next three layers ($j\le3$) use a one-step extrapolation $p_j = p_{j-1} + \Delta p_{j-1}$, because the history is not yet long enough for the full formula. From the fifth layer on, a four-term multistep predictor combines the value four layers back with a weighted blend of the last three derivatives,

$$
p_j^{\rm pred} = \frac{3\,p_{j-4} + 8\,\Delta p_{j-1} - 4\,\Delta p_{j-2} + 8\,\Delta p_{j-3}}{3}.
$$

These are the Kurucz/ATLAS multistep coefficients; we do not derive them here but transcribe them from ATLAS. They play the same role as an explicit Adams-type predictor on this smooth log-log grid, reaching back several layers because the integrand is smooth enough there that a long stencil pays off. One useful sanity check is that the predictor is exact for a linear $p(\ln\tau)$ — which is precisely the cold-start limit, where $\kappa\equiv1$ makes $\ln P$ linear in $\ln\tau$.

**Corrector.** With a trial $p_j$ in hand, evaluate the total pressure $P_{{\rm total},j}=e^{p_j}$, subtract the radiation (and turbulent) pressure to get the gas pressure, look up the opacity ($\kappa_j=1$ on the cold start), and form the derivative $\Delta p_j = (g/\kappa_j)(\tau_j/P_{{\rm total},j})\,\Delta\ln\tau$. A corrector formula (analogous integer weights) then produces a refined estimate $p_j^{\rm new}$ from $p_j$ and the freshly computed derivative, and we average the two, $p_j \leftarrow \tfrac12(p_j^{\rm new}+p_j)$, iterating until the change is below $5\times10^{-5}$.

One ordering detail is what makes this reproduce the reference *to the bit*, and it is worth stating plainly. The code **evaluates and stores** $P_{\rm total}$, the gas pressure, the opacity, and the derivative from the current trial $p_j$ *first*, and only *then* tests for convergence. So on the iteration where the loop decides it is done, the stored pressure is the trial value at the *start* of that iteration — not a further-refined corrector. (If the loop runs more than once, that trial value is itself the averaged corrector from the previous iteration; the corrector that would have nudged it once more is computed but never applied to the stored value.) Matching this "evaluate-then-check" order is the difference between agreeing to five decimals and agreeing to all of them.""")

md(r"""We implement this in two cells. The first, `ttaup`, is the **set-up wrapper**: it allocates the output arrays (opacity, total pressure, gas pressure), forms the constant log-tau step, zeroes the rolling history, and seeds the boundary opacity at the top layer ($\kappa_0=0.1$, unless a positive surface radiation pressure asks for a smaller value). It then hands everything to the per-layer loop.

The boundary guard `if prad[0] > 0.0` deserves a word, because the routine is natively written to receive the *absolute* radiation pressure (where `prad[0] > 0.0` and the guard sets a tighter seed). For this cold-start benchmark we deliberately pass the pre-subtracted increment array, which forces `prad[0]` to be exactly zero by construction — and that is the point: a zero `prad[0]` deactivates the guard, replicating ATLAS's behaviour of skipping the boundary correction on the very first pass and falling back to the plain $\kappa_0=0.1$ seed. The integrator is **sequential in the layer index** — the history couples each layer to the previous few — so it cannot be vectorised across layers, and that is why it is a Python loop rather than an array expression.""")

code(r'''def ttaup(t, tau, prad, pturb, grav):
    """Set up arrays + boundary seed for the log-pressure hydrostatic integration (Kurucz TTAUP)."""
    n = t.size

    # output arrays, filled layer by layer in the loop
    abstd  = np.zeros(n)            # opacity kappa_Ross at each layer (== 1 on the cold start)
    ptotal = np.zeros(n)           # total pressure  P_total = exp(log P)
    pgas   = np.zeros(n)           # gas pressure    P_gas   = P_total - P_rad - P_turb
    dlg_tau = np.log(tau[1] / tau[0]) if n > 1 else 0.0   # constant log-tau step (= ln10 * 0.125)

    # rolling history of log P (plog1..plog4) and of the derivative dplog (dplog1..dplog3)
    plog1 = plog2 = plog3 = plog4 = 0.0
    dplog1 = dplog2 = dplog3 = 0.0

    # seed opacity at the top layer (kappa_0 = 0.1) to set the boundary pressure
    abstd[0] = 0.1
    if prad[0] > 0.0:              # warm-restart guard; prad[0]==0 on the cold start, so no-op here
        abstd[0] = min(0.1, grav * tau[0] / max(prad[0], 1e-300) / 2.0)

    return _ttaup_loop(t, tau, prad, pturb, grav, n, abstd, ptotal, pgas, dlg_tau,
                       plog1, plog2, plog3, plog4, dplog1, dplog2, dplog3)''')

md(r"""The second cell is the **engine**: the per-layer loop. It has to stay in one cell because it is a single function — the layer-to-layer history makes it inherently sequential, so it cannot be split or vectorised — but its three parts are marked by comment banners. For each layer $j$ it runs the **predictor** (extrapolate $\ln P$ from the history), then the **corrector** inner loop (the `evaluate-then-check` ordering: store $P_{\rm total}$, gas pressure, opacity, and derivative *first*, then test convergence), then shifts the history forward by one. Read the comment banners as a map of the algorithm above.""")

code(r'''def _ttaup_loop(t, tau, prad, pturb, grav, n, abstd, ptotal, pgas, dlg_tau,
                plog1, plog2, plog3, plog4, dplog1, dplog2, dplog3):
    for j in range(n):

        # ---- PREDICTOR: extrapolate log P from the history ----
        if j == 0:
            plog = np.log(max(grav / abstd[0] * tau[0], 1e-300))     # boundary: kappa_0 = 0.1
        elif j <= 3:
            plog = plog1 + dplog1                                     # short history: one-step
        else:
            plog = (3.0*plog4 + 8.0*dplog1 - 4.0*dplog2 + 8.0*dplog3) / 3.0   # 4-term multistep

        # ---- CORRECTOR loop: EVALUATE the stored values, THEN check convergence ----
        error, dplog, itn = 1.0, 0.0, 1
        while True:
            # evaluate and store pressure, gas pressure, opacity, and derivative from the trial plog
            plog = min(plog, 709.78)                       # guard exp() against overflow
            ptotal[j] = np.exp(plog)                       # total pressure at this layer
            pgas[j] = ptotal[j] + (prad[0] - prad[j]) - pturb[j]     # gas = total - P_rad run - P_turb
            if pgas[j] <= 0.0:                             # non-physical: clamp and stop this layer
                pgas[j] = 1e-30; abstd[j] = 0.1; break
            abstd[j] = 1.0                                 # cold-start opacity (empty ROSSTAB -> 1)
            dplog = grav / abstd[j] * tau[j] / ptotal[j] * dlg_tau   # d(log P) across one log-tau step

            # only now test convergence -- the stored values keep this trial plog
            itn += 1
            if itn > 1000 or error <= 5.0e-5:              # converged (or out of iterations): STOP
                break                                       # -> STORED values use this trial plog

            # corrector estimate, using the just-computed dplog:
            if j == 0:
                pnew = np.log(max(grav / abstd[j] * tau[j], 1e-300))
            elif j <= 3:
                pnew = (plog + 2.0*plog1 + dplog + dplog1) / 3.0
            else:
                pnew = (126.0*plog1 - 14.0*plog3 + 9.0*plog4
                        + 42.0*dplog + 108.0*dplog1 - 54.0*dplog2 + 24.0*dplog3) / 121.0
            error = abs(pnew - plog)                       # convergence test for the NEXT iteration
            plog = 0.5 * (pnew + plog)                     # average predictor and corrector

        # ---- shift the history forward one layer ----
        plog4, plog3, plog2, plog1 = plog3, plog2, plog1, plog
        dplog3, dplog2, dplog1 = dplog2, dplog1, dplog
    return abstd, ptotal, pgas''')

# ── run it ──────────────────────────────────────────────────────────────────
md(r"""## Running the integrator

With the temperature, the radiation pressure, and the opacity convention in hand, we run the integrator and form the two output quantities the structure carries: the **gas pressure** (already done inside the loop) and the **column mass**.

The gas-pressure line in the loop reads `pgas = ptotal + (prad[0] - prad[j]) - pturb`, and the `+ prad[0]` is worth understanding rather than memorising. What the integrator actually accumulates in `ptotal` is the column weight $g\,m = P_{\rm total}(\tau) - P_{\rm total}(0)$ — the pressure relative to the top boundary. Since the boundary carries no gas, $P_{\rm total}(0)$ is exactly the surface radiation pressure $P_{\rm rad}(0)$. To recover the proper gas pressure $P_{\rm gas}(\tau) = P_{\rm total}(\tau) - P_{\rm rad}(\tau)$ the code must therefore add the surface term back before subtracting the local one, which is precisely `ptotal + P_rad(0) - P_rad(tau)`. On this cold start `prad[0]` is zero, so the line reduces to the plain $P_{\rm gas}=P_{\rm total}-P_{\rm rad}$, but the algebra is general. The column mass is then

$$
m = \frac{P_{\rm total}}{g},
$$

the weight per unit area of everything above — the depth variable the `.atm` file is tabulated against and the one the opacity and transfer lectures integrate over.""")

code(r'''# run the predictor-corrector integrator: opacity, total pressure, gas pressure
abstd, ptotal, P_gas = ttaup(T, tau, prad, pturb, g_cgs)

# column mass m = total pressure / gravity (Kurucz's RHOX)
RHOX = ptotal / g_cgs                          # [g cm^-2]

print(f"P_gas:  top = {P_gas[0]:.4e}   bottom = {P_gas[-1]:.4e} dyn/cm^2")
print(f"RHOX:   top = {RHOX[0]:.4e}   bottom = {RHOX[-1]:.4e} g/cm^2")''')

# ── benchmark ───────────────────────────────────────────────────────────────
md(r"""## Benchmark: machine precision

Now the comparison the first lecture deferred. We check the gas pressure and the column mass against the reference grey structure, layer by layer. The first lecture's one-line estimate $P=g\tau$ matched these to $\sim2\times10^{-5}$; reproducing the exact predictor-corrector — in log pressure, with the evaluate-then-check ordering, carrying the radiation-pressure correction — closes that gap to the last bit. The cell runs the same `check` helper on all four arrays — the two recapped ones plus the two residuals — then aggregates: `worst` takes the largest relative difference across them, and `allbit` is `True` only when every array is bit-identical to the reference.""")

code(r'''print("grey model atmosphere vs reference/L1.npz:")

# the two recapped arrays (matched in L1) and the two residuals L1 deferred
e_tau  = check("grey_tau",  tau,   REF["grey_tau"])    # grid        (recap, matched in L1)
e_T    = check("grey_T",    T,     REF["grey_T"])      # temperature (recap, matched in L1)
e_pgas = check("grey_pgas", P_gas, REF["grey_pgas"])   # gas pressure -- the residual L1 deferred
e_rhox = check("grey_rhox", RHOX,  REF["grey_rhox"])   # column mass  -- the residual L1 deferred

# aggregate: worst relative diff, and whether every array is bit-identical
worst = max(e_tau, e_T, e_pgas, e_rhox)
allbit = all(np.array_equal(a, REF[k]) for a, k in
             [(tau,"grey_tau"), (T,"grey_T"), (P_gas,"grey_pgas"), (RHOX,"grey_rhox")])
print(f"\nworst max|rel| over all four arrays = {worst:.2e}")
print(f"all four arrays bit-exact = {allbit}")''')

md(r"""**Machine precision.** All four arrays — optical depth, temperature, gas pressure, column mass — are **bit-for-bit identical** to the reference (relative difference exactly zero). The first lecture's $P=g\tau$ was the exact analytic $\kappa\equiv1$, zero-boundary *total*-pressure solution; what we have now is the exact reproduction of the production code's *discrete* hydrostatic integration of the same equation, with its boundary convention and radiation-pressure treatment — and it is what the production code starts its iteration from. We have built the grey model atmosphere of the Sun, to the bit, from two numbers.""")

md(r"""How much did the discrete integrator move the answer from the one-line estimate? Let us look directly: plot the structure, and the fractional difference between the analytic $P=g\tau$ and the integrated $P_{\rm gas}$.""")

code(r'''P_analytic = g_cgs * tau                              # the first lecture's one-line estimate
fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))

# left: the grey solar structure -- gas pressure and column mass vs log tau
ax[0].plot(np.log10(tau), np.log10(P_gas), color="C0", lw=1.6,
           label=r"$P_{\rm gas}$  [dyn cm$^{-2}$]")
ax[0].plot(np.log10(tau), np.log10(RHOX),  color="C3", lw=1.2, ls="--",
           label=r"$m$ (column mass)  [g cm$^{-2}$]")
ax[0].set_xlabel(r"$\log_{10}\tau$"); ax[0].set_ylabel(r"$\log_{10}$ value (each in its own cgs unit)")
ax[0].set_title("Grey solar structure"); ax[0].legend(loc="upper left")

# right: fractional gap between the one-line estimate and the discrete integration
resid = np.abs(P_analytic - P_gas) / P_gas             # |g*tau  -  P_gas| / P_gas
ax[1].semilogy(np.log10(tau), resid, color="C2", lw=1.4)
ax[1].axhline(2e-5, color="0.6", ls=":", lw=1.0)       # the ~2e-5 level L1 reported
ax[1].set_xlabel(r"$\log_{10}\tau$")
ax[1].set_ylabel(r"$|g\tau - P_{\rm gas}|\,/\,P_{\rm gas}$")
ax[1].set_title(r"where the one-line $P \approx (g/\kappa)\tau$ differed")

fig.tight_layout(); plt.show()
print(f"P=g*tau vs P_gas: max frac diff = {resid.max():.2e}  (this is the gap we just closed)")''')

md(r"""The left panel is the grey solar atmosphere — gas pressure and column mass climbing nine decades from the thin top layers to the deep base, almost a straight line in log-log because $\kappa\equiv1$ makes $P\approx(g/\kappa)\tau$ a power law. The right panel shows where the one-line estimate differed from the stored gas pressure: a fraction of $\sim10^{-5}$, largest in the upper layers where the radiation-pressure subtraction and the multistep integration's start-up depart most from the bare $g\tau$. That gap is now zero — not because the physics changed, but because we integrated the same equation the way the reference does, with its boundary and radiation-pressure conventions.""")

# ── forward look ────────────────────────────────────────────────────────────
md(r"""## What this atmosphere is not yet: radiative equilibrium

We have a structure that satisfies hydrostatic equilibrium — but only with a **placeholder opacity** $\kappa\equiv1$ and a **grey temperature law** that assumed wavelength-independent opacity. Neither is true of a real star. Lecture 10 takes the second structural constraint, **radiative equilibrium**: in a purely radiative atmosphere the radiative flux is constant with depth (no energy created or destroyed in the photosphere); where convection sets in, the conserved quantity is instead the sum of the radiative and convective fluxes. The grey temperature does not satisfy this once the opacity is wavelength-dependent, so the temperature is **corrected**, layer by layer, until the emergent flux matches $\sigma T_{\rm eff}^4$ and the flux divergence vanishes. Feeding that corrected temperature back into this hydrostatic integration — now with the *real* Rosseland-mean opacity from the equation of state instead of $\kappa\equiv1$ — and iterating the two to convergence is how ATLAS builds a self-consistent model atmosphere.

That closes the loop the whole course has been tracing: **parameters $\to$ atmosphere $\to$ spectrum**. The first lecture took the atmosphere as given and we spent the synthesis half (Lectures 1-8) turning it into a spectrum to machine precision; this lecture began the inverse half by building the structure from $T_{\rm eff}$ and $\log g$; and Lecture 10 makes that structure self-consistent. With both halves in hand, a star's two numbers become its spectrum — from the standard grey-atmosphere approximations and the ATLAS numerical conventions, to the bit.""")

# ── synthesis ───────────────────────────────────────────────────────────────
md(r"""## Synthesis

A model atmosphere is the run of temperature, pressure, and density with depth, fixed by hydrostatic and radiative equilibrium. This lecture built the hydrostatic half. Hydrostatic equilibrium in optical depth is $dP/d\tau = g/\kappa$, equivalently $P_{\rm total} = g\,m$ for column mass $m$ (up to the top-boundary constant) — the total pressure is the weight per area of the overlying gas. The grey/Hopf law supplies the temperature, and on the cold start the opacity is the crude placeholder $\kappa\equiv1$ (the empty Rosseland table), which gives the first-guess structure ATLAS then refines. Integrating the balance in **log pressure** with a **predictor-corrector** multistep — and reproducing the production code's evaluate-then-check ordering and its radiation-pressure subtraction — gives the gas pressure and column mass to machine precision, sharpening the first lecture's one-line $P=g\tau$ from one part in $10^{5}$ to the last bit.""")

md(r"""## Summary

- **Hydrostatic equilibrium** in optical depth is $dP/d\tau = g/\kappa$; integrating it against the column mass $m$ gives the clean identity $P_{\rm total} = g\,m$ (up to the top-boundary constant).
- The **grey/Hopf temperature** $T(\tau) = T_{\rm eff}[\tfrac34(0.710 + \tau - 0.1331\,e^{-3.4488\tau})]^{1/4}$ is the input to the pressure integration and sets the (tiny, but carried) radiation pressure $P_{\rm rad} = 2.521\times10^{-15}\max(T^4, \tfrac12 T_{\rm eff}^4)$.
- The grid is the ATLAS12 default: **80 layers**, $\log\tau$ from $-6.875$ in steps of **$0.125$ dex**; the **cold start** uses a placeholder opacity $\kappa\equiv1$ because the real opacity needs the structure that is being solved for.
- The hydrostatic equation is integrated in **log pressure** (many decades, smoothest integrand) by a **predictor-corrector** multistep, with the gas pressure recovered as $P_{\rm gas} = P_{\rm total} - P_{\rm rad}$ and the column mass as $m = P_{\rm total}/g$.
- The rebuilt structure matches the reference **bit-for-bit** (relative difference exactly zero) in temperature, gas pressure, and column mass — the grey model ATLAS12 starts its iteration from.""")

md(r"""## Practice exercises

**1. The boundary seed.** The top layer is integrated with $\kappa_0 = 0.1$ rather than the $\kappa\equiv1$ used everywhere else. Change the seed to $1.0$ and recompute. Which array moves, and by how much at $\tau = \tau_0$? Explain why the seed sets only the boundary value of the integration and washes out within a few layers (look at how quickly the predictor history forgets the first point).

**2. A hotter, lower-gravity star.** Rebuild the grey atmosphere for an A-type dwarf, $T_{\rm eff} = 9000\,\mathrm{K}$, $\log g = 4.0$. Overplot its $P_{\rm gas}(\tau)$ on the Sun's. Why is the gas pressure lower at fixed $\tau$, given $P_{\rm total} = g\,m$? And at what $T_{\rm eff}$ does the radiation-pressure subtraction $P_{\rm rad}$ become a percent-level correction to $P_{\rm gas}$ in the deep layers?

**3. Why log pressure.** Re-derive the per-step derivative $\Delta p = (g/\kappa)(\tau/P_{\rm total})\,\Delta\ln\tau$ from $dP/d\tau = g/\kappa$ and $p = \ln P$. Then integrate the cold-start equation analytically with $\kappa\equiv1$ to confirm $P = g\tau$, and explain why $\log P$ is exactly linear in $\log\tau$ in that limit — and therefore why a low-order multistep is so accurate here.

**4. The evaluate-then-check ordering.** Modify the corrector loop to apply one more corrector step *after* the convergence test fires (i.e. update the stored pressure with the final corrector). Recompute the benchmark. How large is the change, and at which layers? This is the difference between agreeing to five decimals and agreeing to all of them — explain why a tiny reordering has a measurable, if small, effect.""")

md(r"""## Further reading

- **Gray, D. F. (2005). *The Observation and Analysis of Stellar Photospheres*, 3rd ed., Cambridge University Press.** Chapters 7-9 on the grey atmosphere, hydrostatic equilibrium, and the temperature structure at the level of this course.
- **Mihalas, D. (1978). *Stellar Atmospheres*, 2nd ed., Freeman.** Chapter 3 on the construction of model atmospheres: hydrostatic and radiative equilibrium, and the iteration that couples them.
- **Hubeny, I. & Mihalas, D. (2014). *Theory of Stellar Atmospheres*, Princeton.** Chapters 12 and 17-18 on the equations of stellar-atmosphere structure and the numerical methods that solve them.
- **Kurucz, R. L. (1970). *ATLAS: A Computer Program for Calculating Model Stellar Atmospheres*, SAO Special Report 309.** The original ATLAS, including the grey cold start (`CALCULATE` card) and the `TTAUP` hydrostatic integrator we reproduce here.
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz*](https://arxiv.org/abs/2603.11693).** The implementation our reference grey atmosphere is computed with.""")

nb = new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
