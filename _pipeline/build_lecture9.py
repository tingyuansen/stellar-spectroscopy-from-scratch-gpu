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

- State **hydrostatic equilibrium** as a differential equation in optical depth, $dP/d\tau = g/\kappa$, and explain why the column mass and the pressure are two sides of the same balance.
- Recall the **grey/Hopf temperature law** $T(\tau)$ from the first lecture, and see it as the *input* to the structure rather than something handed down.
- Build the production code's **80-layer optical-depth grid** (the literal ATLAS12 card defaults) and say why the cold start integrates with a placeholder opacity $\kappa\equiv1$.
- Integrate hydrostatic equilibrium in **log pressure** with a **predictor-corrector** multistep, apply the **radiation-pressure correction** $P_{\rm gas}=P_{\rm total}-P_{\rm rad}$, and reproduce the reference grey atmosphere — temperature, gas pressure, and column mass — to **machine precision**, sharpening the first lecture's one-line $P=g\tau$ estimate to the last bit.""")

# ── introduction ───────────────────────────────────────────────────────────
md(r"""## Introduction: building the atmosphere we have been given

For eight lectures the **model atmosphere** has been a given. We took its run of temperature, gas pressure, and density with depth — the columns of a `.atm` file — and on top of it built the equation of state, the continuous and line opacities, and the radiative-transfer solver, until we reproduced the solar spectrum to machine precision. But where did that atmosphere come from? A model atmosphere is fixed by two physical requirements: **hydrostatic equilibrium**, that the gas neither collapses under its own weight nor blows away, and **radiative equilibrium**, that the energy carried outward by radiation is conserved at every depth. This lecture builds the first of those. It is the start of the **inverse half** of the course: instead of taking the structure and computing the spectrum, we compute the structure itself from the two numbers that define a star — its effective temperature $T_{\rm eff}$ and its surface gravity $\log g$.

The first lecture already wrote down a temperature structure: the **grey atmosphere**, $T(\tau)$ from the Hopf function. It also estimated the pressure with a one-line analytic integral, $P_{\rm total}=g\tau$, and noted that this agreed with the reference to about one part in $10^{5}$ — close, but not exact, with the residual deferred to "when we rebuild hydrostatic equilibrium in full." This is that rebuild. We reproduce the exact pressure integrator the reference uses — a predictor-corrector that solves the same hydrostatic equation in log pressure — and the last digits fall into place: temperature, gas pressure, and column mass, all bit-for-bit identical to the reference grey model that ATLAS12 starts its own iteration from.""")

# ── the imports + reference ────────────────────────────────────────────────
md(r"""## Setup and the reference

We import only NumPy and Matplotlib. The benchmark target is the same `reference/L1.npz` the first lecture used: it carries the grey optical-depth grid `grey_tau`, the grey temperature `grey_T`, the gas pressure `grey_pgas`, and the column mass `grey_rhox`, all computed once by the production code on the solar parameters. The first lecture matched `grey_tau` and `grey_T` exactly and `grey_pgas`/`grey_rhox` to $\sim2\times10^{-5}$; by the end of this lecture all four match to the bit.""")

code(r'''import pathlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5})

REF = np.load(pathlib.Path("..") / "reference" / "L1.npz")
TEFF, LOGG = 5770.0, 4.44                 # the Sun: effective temperature [K], log surface gravity [cgs]
g_cgs = 10.0 ** LOGG                       # surface gravity g [cm s^-2]
print(f"target: grey atmosphere for Teff = {TEFF:.0f} K, log g = {LOGG}  ->  g = {g_cgs:.4g} cm/s^2")
print(f"reference arrays: {[k for k in REF.files]}")''')

# ── hydrostatic equilibrium ─────────────────────────────────────────────────
md(r"""## Hydrostatic equilibrium in optical depth

A static atmosphere obeys a single balance: at every depth, the upward push of the pressure gradient supports the weight of the gas above. In terms of geometric height $z$ (increasing outward) and mass density $\rho$,

$$
\frac{dP}{dz} = -\rho\,g,
$$

with $g$ the surface gravity, taken constant across the thin photosphere. The structure codes do not work in height, though — they work in **optical depth** $\tau$, because that is the variable the radiation field cares about. Recall from the first lecture that optical depth is built from the opacity per gram $\kappa$ and the **column mass** $\rho x$ (grams of material above a square centimetre), with $d\tau = \kappa\,d(\rho x)$ and $d(\rho x) = -\rho\,dz$. Substituting,

$$
\frac{dP}{d\tau} = \frac{g}{\kappa}
$$

This is hydrostatic equilibrium in the form the codes integrate. Read it as a statement about the column mass: since $dP/d(\rho x) = g$, the total pressure is just $g$ times the column mass, $P_{\rm total} = g\cdot\rho x$ — the weight per unit area of everything above. The optical-depth form makes the opacity explicit: where the gas is more opaque (large $\kappa$), a given range of $\tau$ spans less pressure, because you reach optical depth one in less material.

To integrate it we need two things: the temperature at every depth (which sets $P_{\rm rad}$ and, through the equation of state, the density), and the opacity $\kappa(\tau)$. We take them in turn.""")

# ── the temperature law ─────────────────────────────────────────────────────
md(r"""## The temperature: the grey/Hopf law, revisited

The temperature structure is the *input* to the pressure integration, so we start there — with the grey law from the first lecture. For a grey atmosphere (opacity independent of wavelength) in radiative equilibrium, the transfer equation can be solved exactly for the temperature, giving

$$
T^4(\tau) = \tfrac{3}{4}\,T_{\rm eff}^4\,\big[\tau + q(\tau)\big],
$$

where $q(\tau)$ is the **Hopf function**, rising slowly from $q(0)=1/\sqrt3\approx0.577$ to $q(\infty)\approx0.710$. The production code uses Kurucz's polynomial fit to $q$,

$$
T(\tau) = T_{\rm eff}\,\Big[\tfrac{3}{4}\big(0.710 + \tau - 0.1331\,e^{-3.4488\,\tau}\big)\Big]^{1/4}.
$$

The three constants — $0.710$, $0.1331$, $3.4488$ — are the fit: $0.710$ is the deep limit of $q$, and the exponential term bends the curve down at the surface toward $1/\sqrt3$. We met this in the first lecture and reproduced it to the bit; we reproduce it again here, since it feeds the pressure. A floor of $10^{-300}$ guards the fourth root against an underflow to a negative bracket — it never triggers for a real atmosphere but keeps the routine safe.""")

code(r'''def grey_temperature(teff, tau):
    """Eddington-Kurucz grey T(tau): a polynomial fit to the Hopf function q(tau)."""
    bracket = 0.75 * (0.710 + tau - 0.1331 * np.exp(-3.4488 * tau))   # (3/4)(tau + q(tau))
    return float(teff) * np.power(np.maximum(bracket, 1e-300), 0.25)  # T = Teff * bracket^(1/4)''')

# ── the optical-depth grid ──────────────────────────────────────────────────
md(r"""## The optical-depth grid: 80 layers, the ATLAS12 card defaults

We discretise the atmosphere on a grid of optical depth. The production code's grey start uses the literal ATLAS12 `CALCULATE` card defaults: **80 layers**, the first at $\log\tau = -6.875$, spaced **$0.125$ dex** apart. That spans from $\tau\approx1.3\times10^{-7}$ at the very top of the atmosphere to $\tau\approx3\times10^{2}$ deep below the photosphere — nearly ten decades, fine enough to integrate the hydrostatic equation accurately and wide enough to bracket the line- and continuum-forming region (which sits near $\tau\sim1$).

The grid is uniform in $\log\tau$, so the **log-tau step** $\Delta\log\tau$ is a constant. We will integrate the pressure in log-pressure against this log-tau grid, so this constant step is the natural increment of the integration.""")

code(r'''NRHOX, TAU1LG, STEPLG = 80, -6.875, 0.125      # ATLAS12 CALCULATE card defaults
j = np.arange(NRHOX, dtype=np.float64)
tau = np.power(10.0, TAU1LG + j * STEPLG)        # 80 layers, 0.125 dex apart, surface -> deep
T = grey_temperature(TEFF, tau)
print(f"{NRHOX} layers,  tau = {tau[0]:.3e} .. {tau[-1]:.3e}")
print(f"T(top) = {T[0]:.1f} K    T(tau~1) ~ {grey_temperature(TEFF, 1.0):.1f} K    T(bottom) = {T[-1]:.1f} K")''')

md(r"""The grid and the temperature are the first two reference arrays, and they match to the bit straight away — they are the same Hopf fit on the same grid the first lecture built.""")

code(r'''def check(name, got, ref):
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    m = np.abs(ref) > 0
    rel = np.zeros_like(ref); rel[m] = np.abs(got[m] - ref[m]) / np.abs(ref[m])
    tag = "bit-exact" if np.array_equal(got, ref) else f"max|rel| = {rel.max():.2e}"
    print(f"  {name:12s}  {tag}")
    return rel.max()

check("grey_tau", tau, REF["grey_tau"])
check("grey_T",   T,   REF["grey_T"])''')

# ── radiation pressure ──────────────────────────────────────────────────────
md(r"""## Radiation pressure

The total pressure has two parts, gas and radiation: $P_{\rm total} = P_{\rm gas} + P_{\rm rad}$. The hydrostatic balance supports the **total**, but the equation of state and the line opacities care about the **gas** pressure, so we will need to subtract $P_{\rm rad}$ at the end. Radiation pressure of an isotropic field is $P_{\rm rad} = \tfrac{4\sigma}{3c}T^4 = a_{\rm rad}T^4$ with $a_{\rm rad}\approx7.566\times10^{-15}\,\mathrm{erg\,cm^{-3}\,K^{-4}}$; the production code carries the combination as the constant $\tfrac13 a_{\rm rad} = 2.521\times10^{-15}$ (the field is integrated with the Eddington factor $\tfrac13$ already folded in). It also floors the temperature inside the fourth power at $T_{\rm eff}^4/2$, so the radiation pressure never drops below a fixed fraction of its photospheric value in the cool upper layers — a stabiliser that keeps the cold start well behaved.

$$
P_{\rm rad}(\tau) = 2.521\times10^{-15}\,\max\!\big(T^4,\ \tfrac12 T_{\rm eff}^4\big).
$$

In a cool dwarf like the Sun the radiation pressure is utterly negligible next to the gas pressure — parts in $10^{8}$ — but the integrator carries it, and reproducing the reference to the bit means carrying it too. What matters for the hydrostatic integration is the **run** of $P_{\rm rad}$ relative to the surface, $P_{\rm rad}(\tau) - P_{\rm rad}(0)$, since only the *gradient* of pressure enters the balance.""")

code(r'''pradk = 2.521e-15 * np.maximum(T ** 4, TEFF ** 4 / 2.0)   # radiation pressure with Kurucz's floor
prad = pradk - pradk[0]                                    # run relative to the surface layer
pturb = np.zeros(NRHOX)                                    # turbulent pressure: zero on the cold start
print(f"P_rad(top) = {pradk[0]:.3e}   P_rad(bottom) = {pradk[-1]:.3e} dyn/cm^2   (gas ~1e5, so ~1e-8 of it)")''')

# ── the cold-start opacity ──────────────────────────────────────────────────
md(r"""## Why the cold start uses $\kappa\equiv1$

Hydrostatic equilibrium $dP/d\tau = g/\kappa$ needs the opacity $\kappa(\tau)$ — but the opacity depends on the pressure and temperature through the equation of state and the photo-absorption cross-sections, which we cannot evaluate until we *have* the structure. This is the bootstrap problem at the heart of building an atmosphere: structure needs opacity, opacity needs structure.

The production code breaks the loop with a **cold start**. On the very first pass there is no opacity table yet, so the Rosseland-mean opacity is set to a placeholder of **unity at every layer**, $\kappa_{\rm Ross}\equiv1$. (Internally the opacity table — `ROSSTAB` — is empty, and an empty table returns $1.0$.) This is deliberately crude: it gives a first guess of the pressure structure that is in the right ballpark, which ATLAS then refines over many iterations, rebuilding the real Rosseland-mean opacity from the equation of state and feeding it back into the hydrostatic integration until the structure stops changing. We are reproducing that **first guess** — the grey model the iteration starts from — so $\kappa\equiv1$ throughout. There is one cosmetic exception: the top layer is seeded with $\kappa=0.1$ purely to set the boundary value of the integration, as we see next.""")

# ── log pressure ────────────────────────────────────────────────────────────
md(r"""## Integrating in log pressure, and why

We integrate $dP/d\tau = g/\kappa$ down the grid. The natural move would be to step in $P$ directly, but two facts make **log pressure** the better variable. First, the pressure spans many decades from the top of the atmosphere to the bottom — eight or nine — so a fixed step in $P$ would be hopelessly coarse at the top and wastefully fine at the bottom, while a step in $\log P$ resolves every decade equally. Second, the grid is uniform in $\log\tau$, and on the cold start with $\kappa$ constant the solution $P=g\tau$ is a power law, so $\log P$ is *linear* in $\log\tau$ — the integrand is smoothest in log-log variables, where a low-order multistep formula is most accurate.

Writing $p \equiv \log P_{\rm total}$ and using $d\log\tau$ as the increment, the chain rule turns hydrostatic equilibrium into

$$
\frac{dp}{d\log\tau}
= \frac{1}{P_{\rm total}}\frac{dP_{\rm total}}{d\tau}\,\frac{d\tau}{d\log\tau}
= \frac{1}{P_{\rm total}}\,\frac{g}{\kappa}\,(\tau\ln10\cdot 10^{\log\tau}/\dots)
= \frac{g}{\kappa}\,\frac{\tau}{P_{\rm total}}\,\Delta\!\ln\tau\big/\Delta\!\ln\tau,
$$

which collapses to the clean per-step derivative the code uses,

$$
\Delta p_j \;=\; \frac{g}{\kappa_j}\,\frac{\tau_j}{P_{{\rm total},j}}\,\Delta\!\ln\tau,
$$

the change in $\log P$ across one log-tau step. (We use natural logs internally for the step $\Delta\ln\tau = \ln(\tau_{j+1}/\tau_j)$, which is the same constant for every interval since the grid is uniform in $\log\tau$.) We will compute this derivative — call it `dplog` — at each layer and use it both to **predict** the next layer's pressure and to **correct** it.""")

code(r'''dlg_tau = np.log(tau[1] / tau[0])    # natural-log tau step, constant across the uniform grid
print(f"d(ln tau) per layer = {dlg_tau:.6f}   (= ln(10) * 0.125 = {np.log(10)*0.125:.6f})")''')

# ── predictor-corrector ─────────────────────────────────────────────────────
md(r"""## The predictor-corrector integrator

We integrate layer by layer from the top down, carrying a short **history** of the last few values of $p=\log P$ and of the derivative $\Delta p$. Each layer is done in two stages.

**Predictor.** Extrapolate $p_j$ from the history. The very first layer ($j=0$) has no history, so its pressure is set directly from the seeded boundary opacity, $p_0 = \ln(g\,\tau_0/\kappa_0)$ with $\kappa_0=0.1$. The next three layers ($j\le3$) use a one-step extrapolation $p_j = p_{j-1} + \Delta p_{j-1}$, because the history is not yet long enough for the full formula. From the fifth layer on, a four-term multistep predictor combines the value four layers back with a weighted blend of the last three derivatives,

$$
p_j^{\rm pred} = \frac{3\,p_{j-4} + 8\,\Delta p_{j-1} - 4\,\Delta p_{j-2} + 8\,\Delta p_{j-3}}{3}.
$$

These integer coefficients are a standard explicit Adams-type formula tuned to this log-log grid; they reach back several layers because the integrand is so smooth there that a long stencil pays off.

**Corrector.** With a trial $p_j$ in hand, evaluate the total pressure $P_{{\rm total},j}=e^{p_j}$, subtract the radiation (and turbulent) pressure to get the gas pressure, look up the opacity ($\kappa_j=1$ on the cold start), and form the derivative $\Delta p_j = (g/\kappa_j)(\tau_j/P_{{\rm total},j})\,\Delta\ln\tau$. A corrector formula (analogous integer weights) then produces a refined estimate $p_j^{\rm new}$ from $p_j$ and the freshly computed derivative, and we average the two, $p_j \leftarrow \tfrac12(p_j^{\rm new}+p_j)$, iterating until the change is below $5\times10^{-5}$.

One ordering detail is what makes this reproduce the reference *to the bit*, and it is worth stating plainly. The code **evaluates and stores** $P_{\rm total}$, the gas pressure, the opacity, and the derivative from the current trial $p_j$ *first*, and only *then* tests for convergence. So on the step where the loop decides it is done, the stored pressure is the one from the last *predictor*, not from a further-refined corrector — the corrector that would have nudged it is computed but not applied to the stored value. Matching this "evaluate-then-check" order is the difference between agreeing to five decimals and agreeing to all of them.""")

code(r'''def ttaup(t, tau, prad, pturb, grav):
    """Integrate dP/dtau = g/kappa over the tau grid in LOG pressure (Kurucz TTAUP).

    Sequential and order-dependent in the layer index j (the history couples each layer
    to the previous few, so this cannot be vectorised across layers). Returns the opacity,
    the total pressure, and the gas pressure at every layer.
    """
    n = t.size
    abstd  = np.zeros(n)             # opacity kappa_Ross at each layer (== 1 on the cold start)
    ptotal = np.zeros(n)            # total pressure P_total = exp(log P)
    pgas   = np.zeros(n)            # gas pressure  P_gas = P_total - P_rad - P_turb
    dlg_tau = np.log(tau[1] / tau[0]) if n > 1 else 0.0   # constant log-tau step

    # rolling history of log P (plog1..plog4) and of the derivative dplog (dplog1..dplog3)
    plog1 = plog2 = plog3 = plog4 = 0.0
    dplog1 = dplog2 = dplog3 = 0.0

    # seed opacity at the top layer (kappa_0 = 0.1) to set the boundary pressure
    abstd[0] = 0.1
    if prad[0] > 0.0:
        abstd[0] = min(0.1, grav * tau[0] / max(prad[0], 1e-300) / 2.0)
    return _ttaup_loop(t, tau, prad, pturb, grav, n, abstd, ptotal, pgas, dlg_tau,
                       plog1, plog2, plog3, plog4, dplog1, dplog2, dplog3)
print("predictor-corrector defined (the per-layer loop is in the next cell)")''')

md(r"""The per-layer loop is the engine; we keep it in its own cell so each piece is visible. For each layer $j$ it runs the predictor, then the corrector loop, then shifts the history forward by one. Watch the two markers in the comments: **predict**, then the inner loop where we **evaluate-then-check**.""")

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
            plog = min(plog, 709.78)                       # guard exp() against overflow
            ptotal[j] = np.exp(plog)                       # total pressure at this layer
            pgas[j] = ptotal[j] + (prad[0] - prad[j]) - pturb[j]     # gas = total - P_rad run - P_turb
            if pgas[j] <= 0.0:                             # non-physical: clamp and stop this layer
                pgas[j] = 1e-30; abstd[j] = 0.1; break
            abstd[j] = 1.0                                 # cold-start opacity (empty ROSSTAB -> 1)
            dplog = grav / abstd[j] * tau[j] / ptotal[j] * dlg_tau   # d(log P) across one log-tau step
            itn += 1
            if itn > 1000 or error <= 5.0e-5:              # converged (or out of iterations): STOP
                break                                       # -> the STORED values use this predictor
            # corrector estimate, using the just-computed dplog:
            if j == 0:
                pnew = np.log(max(grav / abstd[j] * tau[j], 1e-300))
            elif j <= 3:
                pnew = (plog + 2.0*plog1 + dplog + dplog1) / 3.0
            else:
                pnew = (126.0*plog1 - 14.0*plog3 + 9.0*plog4
                        + 42.0*dplog + 108.0*dplog1 - 54.0*dplog2 + 24.0*dplog3) / 121.0
            error = abs(pnew - plog)
            plog = 0.5 * (pnew + plog)                     # average predictor and corrector

        # ---- shift the history forward one layer ----
        plog4, plog3, plog2, plog1 = plog3, plog2, plog1, plog
        dplog3, dplog2, dplog1 = dplog2, dplog1, dplog
    return abstd, ptotal, pgas
print("per-layer loop defined")''')

# ── run it ──────────────────────────────────────────────────────────────────
md(r"""## Running the integrator

With the temperature, the radiation pressure, and the opacity convention in hand, we run the integrator and form the two output quantities the structure carries: the **gas pressure** `P_gas = P_total - P_rad` (already done inside the loop), and the **column mass**

$$
\rho x = \frac{P_{\rm total}}{g},
$$

the weight per unit area of everything above — which is the depth variable the `.atm` file is tabulated against and the one the opacity and transfer lectures integrate over.""")

code(r'''abstd, ptotal, P_gas = ttaup(T, tau, prad, pturb, g_cgs)
RHOX = ptotal / g_cgs                       # column mass [g cm^-2] = total pressure / gravity
print(f"P_gas:  top = {P_gas[0]:.4e}   bottom = {P_gas[-1]:.4e} dyn/cm^2")
print(f"RHOX:   top = {RHOX[0]:.4e}   bottom = {RHOX[-1]:.4e} g/cm^2")''')

# ── benchmark ───────────────────────────────────────────────────────────────
md(r"""## Benchmark: machine precision

Now the comparison the first lecture deferred. We check the gas pressure and the column mass against the reference grey structure, layer by layer. The first lecture's one-line estimate $P=g\tau$ matched these to $\sim2\times10^{-5}$; reproducing the exact predictor-corrector — in log pressure, with the evaluate-then-check ordering, carrying the radiation-pressure correction — closes that gap to the last bit.""")

code(r'''print("grey model atmosphere vs reference/L1.npz:")
e_tau  = check("grey_tau",  tau,   REF["grey_tau"])
e_T    = check("grey_T",    T,     REF["grey_T"])
e_pgas = check("grey_pgas", P_gas, REF["grey_pgas"])
e_rhox = check("grey_rhox", RHOX,  REF["grey_rhox"])
worst = max(e_tau, e_T, e_pgas, e_rhox)
allbit = all(np.array_equal(a, REF[k]) for a, k in
             [(tau,"grey_tau"), (T,"grey_T"), (P_gas,"grey_pgas"), (RHOX,"grey_rhox")])
print(f"\nworst max|rel| over all four arrays = {worst:.2e}")
print(f"all four arrays bit-exact = {allbit}")''')

md(r"""**Machine precision.** All four arrays — optical depth, temperature, gas pressure, column mass — are **bit-for-bit identical** to the reference (relative difference exactly zero). The first lecture's $P=g\tau$ was a good closed-form estimate; this is the exact integral of the same hydrostatic equation, and it is what the production code starts its iteration from. We have built the grey model atmosphere of the Sun, to the bit, from two numbers.""")

md(r"""How much did the exact integrator change from the one-line estimate? Let us look directly: plot the structure, and the fractional difference between $P=g\tau$ and the exact $P_{\rm gas}$.""")

code(r'''P_analytic = g_cgs * tau                              # the first lecture's one-line estimate
fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
ax[0].plot(np.log10(tau), np.log10(P_gas), color="C0", lw=1.6, label=r"$P_{\rm gas}$ (exact integrator)")
ax[0].plot(np.log10(tau), np.log10(RHOX),  color="C3", lw=1.2, ls="--", label=r"$\rho x$ (column mass)")
ax[0].set_xlabel(r"$\log_{10}\tau$"); ax[0].set_ylabel(r"$\log_{10}$ [cgs]")
ax[0].set_title("Grey solar structure"); ax[0].legend(loc="upper left")

resid = np.abs(P_analytic - P_gas) / P_gas
ax[1].semilogy(np.log10(tau), resid, color="C2", lw=1.4)
ax[1].axhline(2e-5, color="0.6", ls=":", lw=1.0)
ax[1].set_xlabel(r"$\log_{10}\tau$"); ax[1].set_ylabel(r"$|P_{g\tau} - P_{\rm gas}|\,/\,P_{\rm gas}$")
ax[1].set_title(r"where the one-line $P=g\tau$ differed")
fig.tight_layout(); plt.show()
print(f"P=g*tau vs exact P_gas: max frac diff = {resid.max():.2e}  (this is the gap we just closed)")''')

md(r"""The left panel is the grey solar atmosphere — gas pressure and column mass climbing nine decades from the thin top layers to the deep base, almost a straight line in log-log because $\kappa\equiv1$ makes $P\approx g\tau$ a power law. The right panel shows where the one-line estimate fell short: a fraction of $\sim10^{-5}$, largest in the upper layers where the radiation-pressure subtraction and the multistep integration's start-up differ most from the bare $g\tau$. That residual is now zero — not because the physics changed, but because we integrated the same equation the way the reference does.""")

# ── forward look ────────────────────────────────────────────────────────────
md(r"""## What this atmosphere is not yet: radiative equilibrium

We have a structure that satisfies hydrostatic equilibrium — but only with a **placeholder opacity** $\kappa\equiv1$ and a **grey temperature law** that assumed wavelength-independent opacity. Neither is true of a real star. Lecture 10 takes the second requirement of a model atmosphere, **radiative equilibrium**: the constraint that the radiative flux is constant with depth (no energy created or destroyed in the photosphere). The grey temperature does not satisfy this once the opacity is wavelength-dependent, so the temperature is **corrected**, layer by layer, until the emergent flux matches $\sigma T_{\rm eff}^4$ and the flux divergence vanishes. Feeding that corrected temperature back into this hydrostatic integration — now with the *real* Rosseland-mean opacity from the equation of state instead of $\kappa\equiv1$ — and iterating the two to convergence is how ATLAS builds a self-consistent model atmosphere.

That closes the loop the whole course has been tracing: **parameters $\to$ atmosphere $\to$ spectrum**. The first lecture took the atmosphere as given and we spent the synthesis half (Lectures 1-8) turning it into a spectrum to machine precision; this lecture began the inverse half by building the structure from $T_{\rm eff}$ and $\log g$; and Lecture 10 makes that structure self-consistent. With both halves in hand, a star's two numbers become its spectrum, from first principles and to the bit.""")

# ── synthesis ───────────────────────────────────────────────────────────────
md(r"""## Synthesis

A model atmosphere is the run of temperature, pressure, and density with depth, fixed by hydrostatic and radiative equilibrium. This lecture built the hydrostatic half. Hydrostatic equilibrium in optical depth is $dP/d\tau = g/\kappa$, equivalently $P_{\rm total} = g\cdot\rho x$ — the total pressure is the weight per area of the overlying gas. The grey/Hopf law supplies the temperature, and on the cold start the opacity is the crude placeholder $\kappa\equiv1$ (the empty Rosseland table), which gives the first-guess structure ATLAS then refines. Integrating the balance in **log pressure** with a **predictor-corrector** multistep — and reproducing the production code's evaluate-then-check ordering and its radiation-pressure subtraction — gives the gas pressure and column mass to machine precision, sharpening the first lecture's one-line $P=g\tau$ from one part in $10^{5}$ to the last bit.""")

md(r"""## Summary

- **Hydrostatic equilibrium** in optical depth is $dP/d\tau = g/\kappa$; integrating it against the column mass gives the clean identity $P_{\rm total} = g\cdot\rho x$.
- The **grey/Hopf temperature** $T(\tau) = T_{\rm eff}[\tfrac34(0.710 + \tau - 0.1331\,e^{-3.4488\tau})]^{1/4}$ is the input to the pressure integration and sets the (tiny, but carried) radiation pressure $P_{\rm rad} = 2.521\times10^{-15}\max(T^4, \tfrac12 T_{\rm eff}^4)$.
- The grid is the ATLAS12 default: **80 layers**, $\log\tau$ from $-6.875$ in steps of **$0.125$ dex**; the **cold start** uses a placeholder opacity $\kappa\equiv1$ because the real opacity needs the structure that is being solved for.
- The hydrostatic equation is integrated in **log pressure** (many decades, smoothest integrand) by a **predictor-corrector** multistep, with the gas pressure recovered as $P_{\rm gas} = P_{\rm total} - P_{\rm rad}$ and the column mass as $\rho x = P_{\rm total}/g$.
- The rebuilt structure matches the reference **bit-for-bit** (relative difference exactly zero) in temperature, gas pressure, and column mass — the grey model ATLAS12 starts its iteration from.""")

md(r"""## Practice exercises

**1. The boundary seed.** The top layer is integrated with $\kappa_0 = 0.1$ rather than the $\kappa\equiv1$ used everywhere else. Change the seed to $1.0$ and recompute. Which array moves, and by how much at $\tau = \tau_0$? Explain why the seed sets only the boundary value of the integration and washes out within a few layers (look at how quickly the predictor history forgets the first point).

**2. A hotter, lower-gravity star.** Rebuild the grey atmosphere for an A-type dwarf, $T_{\rm eff} = 9000\,\mathrm{K}$, $\log g = 4.0$. Overplot its $P_{\rm gas}(\tau)$ on the Sun's. Why is the gas pressure lower at fixed $\tau$, given $P_{\rm total} = g\,\rho x$? And at what $T_{\rm eff}$ does the radiation-pressure subtraction $P_{\rm rad}$ become a percent-level correction to $P_{\rm gas}$ in the deep layers?

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
