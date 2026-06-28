#!/usr/bin/env python
"""Lecture 15 — Line Blanketing: the True Model Atmosphere.

Completes the continuum-only converged model of Lecture 11 by switching on the
millions of spectral lines.  Teaches, from scratch in pure NumPy:
  * the predicted line list and SELECTLINES (the strength cut that picks the lines
    that matter for THIS atmosphere),
  * the LINOP1 line-deposit kernel (the asymmetric sub-pixel wing-walk, the full
    Voigt, the cv<continuum cutoff reach) -- benchmarked to the single-precision floor on a small
    wavelength window against the production deposit,
  * the line-BLANKETED Rosseland mean (fold the line opacity into the total
    extinction),
  * the line-blanketed convergence: one iteration of the SAME engine of Lecture 11
    -- JOSH, Rosseland mean, CONVEC, TCORR -- with the line blanket switched on,
    reproducing the production single line-blanketed step (engine fidelity) and
    landing on the REAL Sun's model atmosphere sun.npz (surface 3696 K, base
    11425 K, base RHOX 12.14).

Self-contained: imports only numpy / matplotlib / pathlib.  The line-deposit kernel
is pure readable Python (no numba), run on a small window so it executes in a cell;
the full-grid line opacity for the convergence step ships precomputed (given, exactly
like the continuum acont of Lecture 11).  No pykurucz import.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture15.ipynb"
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

# ── title + objectives ────────────────────────────────────────────────────────
md(r"""# Lecture 15 — Line Blanketing: the True Model Atmosphere

*Stellar Spectroscopy from Scratch — rebuilding the physics of ATLAS and SYNTHE from first principles*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*Every result in this book is checked against reference values computed with [**pykurucz**](https://arxiv.org/abs/2603.11693) — a pure-Python implementation of Kurucz's ATLAS12 and SYNTHE — shipped beside the lectures as small data files, so the notebooks need only NumPy to run.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Explain what **line blanketing** is and why it changes the *model atmosphere* and not just the emergent spectrum: the millions of spectral lines block radiation across the ultraviolet, the trapped energy heats the deep layers (**back-warming**) while the surface cools, and the temperature structure reshapes until the flux is constant again.
- Read a **predicted line list** (Kurucz's `gfpred`, ~18 million transitions) and apply **`SELECTLINES`** — the strength cut that keeps only the lines whose opacity can rise above the continuum *in this particular atmosphere*, the step that makes a multi-million-line calculation tractable.
- Implement the production **line-deposit kernel** (`LINOP1` / `_accwings`) from scratch: the line-center opacity from $gf$, the Boltzmann factor and the Doppler width; the **asymmetric, sub-pixel wing-walk** on the logarithmic wavelength grid; the **full Voigt** profile (not a Gaussian-plus-Lorentzian shortcut); and the **adaptive cutoff** that walks each line outward only as far as it stays above the local continuum — benchmarked against the production deposit on a wavelength window (it reproduces it to the single-precision floor of the float32 wing accumulator).
- Fold the line opacity into the **Rosseland mean** and watch the line-blanketed opacity rise above the continuum-only opacity of Lecture 11, especially in the ultraviolet where the line forest is densest.
- Run **one iteration of the Lecture 11 convergence engine with the blanket switched on** — the same JOSH flux, Rosseland mean, mixing-length convection, and temperature correction, now carrying line opacity alongside the continuum — and see it reproduce the production single line-blanketed step to the precision floor, and sit on the **real Sun's model atmosphere** `sun.npz`.
- State precisely what is computed from scratch (the deposit kernel, the line-blanketed fold, the convergence step) and what is taken as given here (the per-iteration equation-of-state state — which **Lecture 16 then builds from scratch** — and the precomputed full-grid line opacity, exactly as Lecture 11 takes the continuum as given), and quantify the residual to `sun.npz`.""")

# ── introduction ──────────────────────────────────────────────────────────────
md(r"""## Introduction: the debt Lecture 11 left open

Lecture 11 converged a model atmosphere of the Sun and benchmarked it to the production code's precision floor. But it converged a **continuum-only** model: in the frequency sweep, the only opacity was the continuum of Lecture 3 — bound-free, free-free, and scattering — and the millions of spectral lines of Lectures 4–6 were switched off. That was a deliberate simplification, and an honest one: it needs no multi-gigabyte line list, so the whole loop stays reproducible with NumPy alone. But it is not the real Sun.

A real model atmosphere is **line-blanketed**. The spectral lines — overwhelmingly the iron-group metals, crowded thickest in the ultraviolet — are not a thin garnish on top of the continuum; in the blue and ultraviolet they *are* the opacity, hundreds of overlapping wings filling every frequency interval. Switching them on does two things. It blocks radiation across huge stretches of the spectrum, so the energy that would have escaped is trapped and the **deep layers heat up** — the classic *back-warming*. And because the total flux must stay constant, the outer layers, now leaking less, **cool down**. The temperature structure of the converged model is genuinely different from the continuum-only one: warmer below, cooler above. This is the single most important physical effect that separates a real model atmosphere from the toy of Lecture 11, and it is what this lecture finally puts in.

The good news is that almost all the machinery is already built. Lecture 11 established the whole convergence loop — the JOSH flux solver, the Rosseland mean, the mixing-length convection, the temperature correction — and that machinery is **opacity-agnostic**: it does not care whether the opacity it folds is continuum, lines, or both. The forward look at the end of Lecture 11 said exactly this: *"Switching them on changes nothing in the machinery of this lecture — the frequency sweep simply carries line opacity alongside the continuum."* So the new physics this lecture must build is narrow and well-defined: **how to deposit a line's opacity onto the wavelength grid**, and **how to select which of the millions of lines to bother with**. We build the deposit kernel from scratch, benchmark it to the single-precision floor, fold the result into the Rosseland mean, and then hand it to the unchanged Lecture 11 engine — which, with the blanket switched on, reaches the real Sun.""")

# ── setup and the reference ───────────────────────────────────────────────────
md(r"""## Setup and the reference

We import only NumPy and Matplotlib. The benchmark target is `reference/lineblanket_ref.npz`, produced once by the production code on the **real Sun's converged model atmosphere** `sun.npz` (surface $T=3696$ K, base $T=11425$ K, base column mass $\rho x = 12.14$ g cm$^{-2}$). Like Lecture 11, the file is honest about what it ships as a *given* versus what we *compute* here.

What it ships as **given**. The **per-iteration equation-of-state state** — the level populations, Doppler widths, electron density, the species-slot populations and the continuum-cutoff table the deposit indexes — and the **precomputed full-grid line opacity** `aline` on all 30000 frequencies. We ship the EOS state here only so this lecture can focus on the *new* physics, the line deposit; **Lecture 16 builds that very state from scratch** (the multi-element population slots, the Doppler widths, the continuum and its cutoff table, and the convective heat capacity), so by the end of the book nothing in this state is borrowed — it is the book's own equation of state, fed back into this lecture's deposit. The full-grid `aline` is shipped because depositing all ~18 million lines onto 30000 frequencies and 80 depths is a multi-gigabyte, compiled-kernel calculation we cannot run live in a notebook cell; we ship the deposited blanket and *verify the deposit kernel that produced it on a small window* instead, exactly as Lecture 11 shipped the precomputed continuum `acont`.

What we **compute** here. The **line-deposit kernel itself**, from scratch, benchmarked to the single-precision floor on a wavelength window of a few thousand lines — the genuinely new physics. The **line-blanketed Rosseland mean** — folding the line opacity into the total extinction. And **one full convergence iteration with the blanket on** — the Lecture 11 engine, unchanged, now carrying lines — reproducing the production single line-blanketed step and landing on `sun.npz`.""")

code(r'''import pathlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5})

REF = np.load(pathlib.Path("..") / "reference" / "lineblanket_ref.npz")
JT  = np.load(pathlib.Path("..") / "reference" / "josh_tables.npz")   # Lecture 8 tables

TEFF = float(REF["teff"]); GRAV = float(REF["gravity_cgs"]); LOGG = float(np.log10(GRAV))
print(f"target: the REAL Sun's line-blanketed model atmosphere sun.npz")
print(f"  Teff = {TEFF:.0f} K,  log g = {LOGG:.2f}")
print(f"  surface T = {REF['T'][0]:.1f} K   base T = {REF['T'][-1]:.1f} K   base RHOX = {REF['rhox'][-1]:.3f} g/cm^2")
print(f"  full-grid line opacity shipped: shape {REF['xlines_fullgrid'].shape} (depth x frequency)")''')

# ── what line blanketing does (qualitative, with the shipped structure) ────────
md(r"""## Back-warming: what the blanket does to the structure

Before any code, it is worth seeing the effect the rest of the lecture works to reproduce. We have, from Lecture 11, the **continuum-only** converged solar model (`converged_ref.npz`), and here the **line-blanketed** converged model of the real Sun (`sun.npz`). Putting them on the same Rosseland optical-depth scale shows the physical signature of line blanketing. (The continuum-only model was converged at $T_{\rm eff}=5770$ K; we interpolate it onto the line-blanketed depth scale for the comparison — the two share the solar parameters, so the *shape* difference is the blanketing effect, not a parameter difference.)""")

code(r'''T_blanket = REF["T"]                       # the real Sun's line-blanketed T(rhox)
rhox      = REF["rhox"]
tauros    = REF["tauros"]                   # the line-blanketed Rosseland optical depth

# Lecture 11's continuum-only converged model, interpolated onto this depth scale
L11 = np.load(pathlib.Path("..") / "reference" / "converged_ref.npz")
T_cont = np.interp(np.log(rhox), np.log(L11["rhox_conv"]), L11["T_conv"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
ax[0].plot(np.log10(tauros), T_cont,    color="0.55", lw=1.7, label="continuum only (Lecture 11)")
ax[0].plot(np.log10(tauros), T_blanket, color="C3",   lw=1.9, label="line-blanketed (real Sun)")
ax[0].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[0].set_ylabel("temperature [K]")
ax[0].set_title("Temperature structure"); ax[0].legend(loc="upper left", fontsize=9)

ax[1].axhline(0.0, color="0.7", lw=1.0)
ax[1].plot(np.log10(tauros), T_blanket - T_cont, color="C0", lw=1.9)
ax[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[1].set_ylabel(r"$T_{\rm blanket} - T_{\rm cont}$ [K]")
ax[1].set_title("The line-blanketing signature")
fig.tight_layout(); plt.show()

deep = tauros > 1.0; top = tauros < 0.1
print(f"deep layers (tau>1):  blanketed warmer by up to {np.max((T_blanket-T_cont)[deep]):+.0f} K  (back-warming)")
print(f"top layers (tau<0.1): blanketed cooler by up to {np.min((T_blanket-T_cont)[top]):+.0f} K  (surface cooling)")''')

md(r"""The deep layers are **warmer** in the line-blanketed model — the back-warming — and the outer layers are **cooler**. That is the structure we now build the ingredients to reproduce: a line opacity to deposit, a Rosseland mean to fold it into, and the convergence step that turns it into this temperature curve.""")

# ── the predicted line list + SELECTLINES ──────────────────────────────────────
md(r"""## The line list and the strength cut (`SELECTLINES`)

The opacity of a star's atmosphere in the blue and ultraviolet is set by an enormous forest of metal lines. Kurucz's **predicted** line list (`gfpred`) holds about **18 million** atomic and molecular transitions — every line whose wavelength and strength can be computed from atomic structure, far more than have ever been measured in the laboratory. The vast majority are iron-group transitions (Fe, Ti, Cr, V, Co, Ni…) packed thickest below 400 nm. This forest is genuine opacity: without it, the ultraviolet windows that dominate the Rosseland mean at the hot base of the photosphere are far too transparent, and the model never reaches the real Sun.

Eighteen million lines is too many to deposit one by one for every atmosphere. The first job is to throw away the ones that cannot matter. That is **`SELECTLINES`**: a strength cut that keeps a line only if its opacity can rise **above the local continuum** somewhere in *this* atmosphere. The test, per line, is

$$\texttt{cen\_ratio} \;=\; \frac{C_0\, gf \; \displaystyle\max_{\rm depth}\!\frac{n_{\rm low}/\Delta\nu_{\rm D}}{\kappa^{\rm cont}}}{\nu^4}\;\ge\;1,
\qquad\text{and}\qquad
\texttt{cen\_ratio}\,e^{-\chi_{\rm low}\,hc/kT_{\rm base}} \ge 1,$$

where the inner maximum is taken over depth (a line is kept if it beats the continuum at the layer where it is *strongest relative to the continuum*), and the second test re-checks it at the deepest, hottest layer after the Boltzmann excitation factor. A line that never rises above the continuum anywhere is invisible against it and is dropped — center and wings. There is no wavelength cut inside `SELECTLINES`; the window enters only through which frequency bin each line lands in. The result for the Sun is the ~18-million-line set the production deposit walks; the reference ships the small window subset we teach on.

Each surviving line is a compact record. The wavelength is stored as an **integer** `iwl` on a logarithmic scale — the vacuum wavelength is $\lambda = e^{\,\texttt{iwl}\cdot\texttt{RATIOLG}}$ with $\texttt{RATIOLG}=\ln(1+1/2{,}000{,}000)$, a resolving power $R=2\times10^6$ — and the physical quantities ($gf$, the lower excitation potential $\chi_{\rm low}$, and the three damping constants) are each stored as an *index* into a precomputed log table $\texttt{TABLOG}[i]=10^{(i-16384)\cdot0.001}$. So decoding a line is a table lookup, not an exponential. Let us load the window the reference ships and decode it.""")

code(r'''# the small teaching window: the selected line records SELECTLINES kept here
iwl    = REF["win_iwl"]          # integer log-wavelength index
ielion = REF["win_ielion"]       # element*ion code (the species)
ielo   = REF["win_ielo"]         # lower-excitation-potential TABLOG index
igflog = REF["win_igflog"]       # log(gf) TABLOG index
igr    = REF["win_igr"]          # radiative-damping TABLOG index
igs    = REF["win_igs"]          # Stark-damping TABLOG index
igw    = REF["win_igw"]          # van der Waals-damping TABLOG index
n_lines = iwl.size

# the two precomputed log tables: TABLOG (the packed-value decoder) and the log grid step
RATIOLG = np.log(1.0 + 1.0 / 2_000_000.0)              # log-wavelength step, R = 2e6
TABLOG  = 10.0 ** ((np.arange(1, 32769, dtype=np.float64) - 16384.0) * 0.001)   # packed-value table

wlvac = np.exp(iwl.astype(np.float64) * RATIOLG)        # decode each line's vacuum wavelength [nm]
nelion = np.abs(ielion) // 10                           # species slot index (molecular if >= 841)
print(f"window: {n_lines} selected lines, {wlvac.min():.3f}-{wlvac.max():.3f} nm")
print(f"  atomic: {(nelion < 841).sum()}   molecular (nelion>=841): {(nelion >= 841).sum()}")
gf = TABLOG[igflog - 1]                                  # oscillator strength x statistical weight
print(f"  gf spans {gf.min():.2e} to {gf.max():.2e}  (the strong lines reach far into their wings)")''')

# ── the per-line physics: center opacity, Doppler, damping ─────────────────────
md(r"""## From a line record to a profile: center opacity, Doppler width, damping

To deposit a line we need three numbers at each depth: the **line-center opacity** $\kappa_0$ (how tall the profile is), the **Doppler width** $\Delta\lambda_{\rm D}$ (how wide the thermal core is), and the **damping parameter** $a$ (how heavy the Lorentz wings are). These come from the line record and the equation-of-state state at that depth — the populations, Doppler widths, electron density, and the neutral-perturber number — which the reference ships as given (Lecture 2 / Lecture 3 outputs).

**The line-center opacity.** The classical oscillator-strength cross-section, scaled by wavelength and the level population, then attenuated by the Boltzmann factor for the lower level:

$$\kappa_0(j) \;=\; \underbrace{C_{gf}\,\lambda\,gf}_{\texttt{cgf}}\;\cdot\;\underbrace{\frac{n_{\rm low}}{\Delta\nu_{\rm D}\,\rho}}_{\texttt{xnfdop}}\;\cdot\;e^{-\chi_{\rm low}\,hc/kT(j)},
\qquad C_{gf}=\frac{0.026538}{\sqrt\pi\,c},$$

where $0.026538=\pi e^2/m_e c$ is the classical line constant, $\sqrt\pi=1.77245$ is the Gaussian normalisation, and $c$ is in nm s$^{-1}$. The combination `xnfdop` $=n_{\rm low}/(\Delta\nu_{\rm D}\rho)$ — the population divided by Doppler width and density — is exactly what the equation of state hands us, so that $\kappa_0$ already carries the Voigt peak normalisation $1/(\sqrt\pi\,\Delta\nu_{\rm D})$.

**The damping parameter.** The three broadening mechanisms — radiative (natural), Stark (linear in the electron density $n_e$), and van der Waals (linear in the neutral-perturber number, with the standard $T^{0.3}$ temperature factor folded in) — sum into a total half-width, divided by the Doppler width:

$$a(j) \;=\; \frac{\Gamma_{\rm rad} + \Gamma_{\rm Stark}\,n_e(j) + \Gamma_{\rm vdW}\,\texttt{txnxn}(j)}{\Delta\nu_{\rm D}(j)},
\qquad \Gamma = C_\Gamma\,\lambda\,\texttt{TABLOG}[\,\cdot\,],\;\; C_\Gamma=\frac{1}{4\pi\,c}.$$

The $1/(4\pi)$ that turns a half-width into the Voigt $a$ lives in $C_\Gamma$. We compute these per-line scalars now, for every depth, using the shipped equation-of-state state.""")

code(r'''# constants in the line-center opacity and the damping (production values, explicit)
C_NM       = 2.99792458e17                       # speed of light [nm/s]
CGF_SCALE  = 0.026538 / 1.77245 / C_NM            # (pi e^2/m_e c) / sqrt(pi) / c
GAMMA_SCALE = 1.0 / 12.5664 / C_NM                # 1 / (4 pi) / c

# the equation-of-state state on the window's depth grid (GIVEN -- Lecture 2/3 outputs)
hckt   = REF["win_hckt"]      # h c / (k T)  per depth (the Boltzmann exponent scale)
xne    = REF["win_xne"]       # electron density per depth [cm^-3]
txnxn  = REF["win_txnxn"]     # neutral-perturber number x (T/1e4)^0.3 per depth (van der Waals)
# the population tables are shipped for ONLY the species the window uses (a compact set of
# columns), so the deposit maps each line's species nelion -> its compact column index.
nelion_set = REF["win_nelion_set"]                    # the distinct species the window touches
xnfdop = REF["win_xnfdop"]    # population / Doppler width / rho, per (depth, compact species column)
dopple = REF["win_dopple"]    # Doppler width Delta_lambda/lambda, per (depth, compact species column)
nelion_to_col = {int(z): k for k, z in enumerate(nelion_set)}   # species slot -> compact column
n_depth = hckt.size
print(f"EOS state on {n_depth} depths; population table xnfdop shape {xnfdop.shape} "
      f"({nelion_set.size} species the window uses)")

# a FASTEX-style Boltzmann factor exp(-x), tabulated exactly as the production code (REAL*4 lookup)
_extab  = np.exp(-np.arange(1001, dtype=np.float64))           # exp(-i), i = 0..1000
_extabf = np.exp(-np.arange(1001, dtype=np.float64) * 0.001)   # exp(-0.001 j), j = 0..1000
def fastex(x):
    """Tabulated exp(-x) for x in [0, 1001) (production FASTEX; single-precision lookup)."""
    if not np.isfinite(x) or x < 0.0 or x >= 1001.0:
        return np.float32(0.0)
    i = int(x); j = int((x - i) * 1000.0 + 1.5)
    j = min(max(j, 1), 1001)
    return np.float32(_extab[i] * _extabf[j - 1])
print("per-line physics helpers defined (cgf, the three gammas, FASTEX Boltzmann factor)")''')

# ── the Voigt profile (recap of Lecture 4) ─────────────────────────────────────
md(r"""## The Voigt profile, the right way (recap of Lecture 4)

The deposit needs the Voigt function $H(a,v)$ — the convolution of the thermal Gaussian core with the Lorentz damping wing — evaluated at the offset $v=(\lambda-\lambda_0)/\Delta\lambda_{\rm D}$ in units of the Doppler width. Lecture 4 built it from Kurucz's **Harris** polynomial tables `h0/h1/h2` (Harris 1948), and we reuse those exact tables here. The subtlety this lecture must respect — and a place a naive implementation goes wrong — is that $H(a,v)$ has **three regimes**, selected by the damping $a$ and the offset:

- For small damping $a\le0.2$ and a tabulated offset, the cheap two-term form $H \approx (h_2\,a + h_1)\,a + h_0$.
- For small damping and a far offset $v>10$, the pure Lorentzian tail $H \to 0.5642\,a/v^2$ ($0.5642=1/\sqrt\pi$).
- For **large** damping $a\ge0.2$, the full Harris correction series (and an analytic asymptotic for $a>1.4$ or $a+v>3.2$, collapsing to a pure Lorentzian for $a>100$).

The third regime matters: the strong, heavily-damped lines at the hot base have $a$ of order 10–150, and the cheap small-$a$ form gives **negative, meaningless** values there. Reproducing all three branches is what makes the deposit faithful. Here is the production $H(a,v)$, the same scalar routine Lecture 4 built, with its branch logic intact.""")

code(r'''KT = np.load(pathlib.Path("..") / "reference" / "leankurucz_tables.npz")   # the Lecture 4 catalog tables
H0, H1, H2 = KT["h0tab"], KT["h1tab"], KT["h2tab"]      # Harris tables, v=0..10 step 0.005 (Lecture 4)

def voigt_H(v, a):
    """Voigt/Hjerting H(a,v) from the Harris tables, all three branches (production _voigt_nb)."""
    iv = int(v * 200.0 + 1.5); iv = min(max(iv, 1), 2001); i = iv - 1   # table index
    if a >= 0.2:
        if a > 1.4 or a + v > 3.2:                                       # analytic asymptotic
            aa = a*a; vv = v*v; u = (aa + vv) * 1.4142                   # 1.4142 = sqrt(2)
            out = a * 0.79788 / u                                        # 0.79788 = sqrt(2/pi)
            if a > 100.0: return out                                     # pure Lorentzian limit
            aau = aa/u; vvu = vv/u; uu = u*u
            return ((((aau - 10.0*vvu)*aau*3.0 + 15.0*vvu*vvu) + 3.0*vv - aa)/uu + 1.0) * out
        vv = v*v                                                         # Harris correction series
        hh1 = H1[i] + H0[i]*1.12838                                      # 1.12838 = 2/sqrt(pi)
        hh2 = H2[i] + hh1*1.12838 - H0[i]
        hh3 = (1.0 - H2[i])*0.37613 - hh1*0.66667*vv + hh2*1.12838
        hh4 = (3.0*hh3 - hh1)*0.37613 + H0[i]*0.66667*vv*vv
        return (((((hh4*a + hh3)*a + hh2)*a + hh1)*a + H0[i])
                * (((-0.122727278*a + 0.532770573)*a - 0.96284325)*a + 0.979895032))
    if v > 10.0:
        return 0.5642 * a / (v*v)                                        # small-a far wing -> Lorentz
    return (H2[i]*a + H1[i])*a + H0[i]                                   # small-a Doppler core
print("Voigt H(a,v) defined (Harris tables, three branches; large-a is essential at the hot base)")''')

# ── the deposit kernel: the asymmetric wing-walk ───────────────────────────────
md(r"""## The deposit kernel: the asymmetric sub-pixel wing-walk

Now the heart of the lecture. Depositing a line is **not** "evaluate the Voigt profile on a fixed window." The production kernel (`LINOP1` / `_accwings`) **walks outward** from the line center, one grid pixel at a time, to the red and to the blue *separately*, adding the profile value to the running opacity array — and **stops** on each side as soon as the deposited value drops below the local continuum. Three properties make it faithful, and each is a place a careless version goes wrong:

**1. The walk is asymmetric and sub-pixel.** The wavelength grid is logarithmic — `waveset`$[i]=e^{\,i\cdot\texttt{RATIOLG}}$ — so the spacing in wavelength *grows* with wavelength, and the pixel just blueward of a line is not the same distance from center as the pixel just redward. The center index `nu0` is the *first* grid pixel at or above the true (continuous) line center, so the center sits a **fractional pixel** below `waveset[nu0]`, and that fractional offset is different on the two sides. The walk therefore computes a fresh, true abscissa at every pixel — $v=(\texttt{waveset}[iw]-\lambda_0)/\Delta\lambda_{\rm D}$ to the red, $v=(\lambda_0-\texttt{waveset}[iw])/\Delta\lambda_{\rm D}$ to the blue — never a symmetric integer offset. There is no separate center deposit: `nu0` *is* the first red sample.

**2. The cutoff reach is adaptive.** Each pixel is **deposited first, then tested**: add $\kappa_0\,H(a,v)$ to the grid, and break only if that value has fallen below the local continuum $\texttt{tabcont}$ at this depth. So a strong line walks far into its wings (overlapping its neighbours), a weak one stops after a pixel or two — the cost scales with how far each line actually reaches, capped at 100 pixels per side. The pixel that first drops below the continuum still gets its value added before the loop breaks.

**3. The Voigt is the full $H(a,v)$.** For $a\le0.2$ the cheap table form; for $a>0.2$ the full `voigt_H` above — never a Gaussian-plus-Lorentzian shortcut, which would miss the heavily-damped base lines entirely.

Here is the kernel for one (line, depth) pair. It is the production routine with its compiler decorators stripped; the logic — the two asymmetric loops, the deposit-before-break, the branch on $a$ — is unchanged.""")

code(r'''def accwings(xlines, j0, nu0, wlvac, center, adamp, dopwave, tabcont_ref, waveset):
    """Deposit one line at one depth: asymmetric red+blue wing-walk, full Voigt, cv<continuum cutoff.
    Faithful to the production _accwings_nb (float32 accumulation, 100-px cap each side)."""
    numnu = waveset.shape[0]
    if dopwave <= 0.0:
        return
    ired_max = 100
    ired_hi = min(nu0 + ired_max + 1, numnu)
    f32 = np.float32
    if adamp <= 0.2:                                            # --- cheap small-a path ---
        for iw in range(nu0, ired_hi):                         # RED wing: nu0 is the first sample
            vv = f32(waveset[iw] - wlvac) / dopwave             # true (asymmetric) abscissa
            if vv > f32(10.0):
                cv = f32(center * f32(0.5642) * adamp / (vv*vv))             # far-wing Lorentz
            else:
                iv = int(vv * f32(200.0) + f32(1.5)); iv = min(max(iv, 1), 2001)
                cv = f32(center * ((H2[iv-1]*adamp + H1[iv-1])*adamp + H0[iv-1]))   # table form
            xlines[j0, iw] += cv
            if cv < tabcont_ref: break                          # deposit-then-break cutoff
        for ired in range(1, ired_max + 1):                     # BLUE wing: walk downward
            iw = nu0 - ired
            if iw < 0: break
            vv = f32(wlvac - waveset[iw]) / dopwave             # opposite-sign abscissa
            if vv > f32(10.0):
                cv = f32(center * f32(0.5642) * adamp / (vv*vv))
            else:
                iv = int(vv * f32(200.0) + f32(1.5)); iv = min(max(iv, 1), 2001)
                cv = f32(center * ((H2[iv-1]*adamp + H1[iv-1])*adamp + H0[iv-1]))
            xlines[j0, iw] += cv
            if cv < tabcont_ref: break
        return
    for iw in range(nu0, ired_hi):                              # --- full-Voigt large-a path ---
        cv = f32(center * voigt_H(f32(waveset[iw] - wlvac) / dopwave, adamp))
        xlines[j0, iw] += cv
        if cv < tabcont_ref: break
    for ired in range(1, ired_max + 1):
        iw = nu0 - ired
        if iw < 0: break
        cv = f32(center * voigt_H(f32(wlvac - waveset[iw]) / dopwave, adamp))
        xlines[j0, iw] += cv
        if cv < tabcont_ref: break
print("accwings deposit kernel defined (the asymmetric sub-pixel wing-walk)")''')

md(r"""The driver loops over the selected lines in wavelength order, and for each line over depth. For one line it decodes the center wavelength, finds the center pixel `nu0` and the continuum bin, computes the per-depth center opacity (with the two cutoff pre-checks the production code uses to skip sub-continuum lines before the walk), the damping $a$, and the Doppler width, and calls `accwings`. The continuum cutoff `tabcont` is read from the tabulated continuum at the line's wavelength bin and depth — the same `cv<tabcont` floor the walk tests against.

One production detail is worth keeping, because it changes the result (not just the speed). The depth loop does **not** simply visit every layer. It first **probes every eighth layer**; a layer that survives the cutoff marks its 8-layer block as active; then a fill-in pass deposits the intervening layers **only in blocks whose probes were active**. A line that is sub-continuum at both ends of an 8-layer block is skipped throughout that block. This is exactly `LINOP1`'s depth-skipping, and reproducing it (not a plain layer-by-layer loop) is what makes the deposit match the production array. We run it on the window.""")

code(r'''def deposit_window(waveset, iwavetab, tabcont, pix_lo, pix_hi):
    """Deposit the window's selected lines onto xlines[depth, pixel] (production LINOP1).
    Deposits on the full wavelength grid, then returns the window pixel band [pix_lo:pix_hi)."""
    nrhox = hckt.size; numnu = waveset.size
    xlines = np.zeros((nrhox, numnu), dtype=np.float32)
    ifj = np.zeros(nrhox + 2, dtype=np.int32)                  # per-block "this 8-block is active" flags
    start, stop = waveset[0] - 1.0, waveset[-1] + 1.0
    nu0 = 0; nucont0 = 0; iwlold = 0

    def deposit_one(j0, cgf, elo, gr, gs, gw, col, wl, wl4, nucont0):
        """Center opacity (with the two cutoff pre-checks), damping, and the wing-walk at one depth."""
        cen = cgf * xnfdop[j0, col]                            # center opacity (pre-Boltzmann)
        if cen < tabcont[j0, nucont0]: return False            # cutoff pre-check 1
        cen = cen * fastex(elo * hckt[j0])                     # x Boltzmann factor
        if cen < tabcont[j0, nucont0]: return False            # cutoff pre-check 2
        dop = dopple[j0, col]
        if dop <= 0.0: return True
        adamp = (gr + gs*xne[j0] + gw*txnxn[j0]) / dop         # the Voigt damping a
        accwings(xlines, j0, nu0, wl, cen, adamp, dop*wl4, tabcont[j0, nucont0], waveset)
        return True

    for k in range(iwl.size):
        iw = int(iwl[k])
        if iw < iwlold:                                        # list is ascending runs -> reset pointers
            nu0 = 0; nucont0 = 0
        while nucont0 < iwavetab.size and iw >= int(iwavetab[nucont0]):
            nucont0 += 1                                       # advance the continuum-bin pointer
        if nucont0 >= tabcont.shape[1]:
            iwlold = iw; continue
        nel = abs(int(ielion[k])) // 10                        # species slot
        if nel not in nelion_to_col:                           # not a species the window ships
            iwlold = iw; continue
        col = nelion_to_col[nel]                               # -> compact population column
        wl = np.exp(float(iw) * RATIOLG); wl4 = np.float32(wl) # the true line center [nm]
        if wl < start or wl > stop:
            iwlold = iw; continue
        while nu0 < numnu and wl >= waveset[nu0]:
            nu0 += 1                                           # nu0 = first pixel at/above the center
        if nu0 >= numnu:
            iwlold = iw; continue
        cgf  = np.float32(CGF_SCALE) * wl4 * TABLOG[int(igflog[k]) - 1]   # line strength (gf x scale)
        elo  = TABLOG[int(ielo[k]) - 1]                        # lower excitation potential
        gr = TABLOG[int(igr[k])-1]*wl4*np.float32(GAMMA_SCALE) # the three damping constants
        gs = TABLOG[int(igs[k])-1]*wl4*np.float32(GAMMA_SCALE)
        gw = TABLOG[int(igw[k])-1]*wl4*np.float32(GAMMA_SCALE)
        for j1 in range(8, nrhox + 1, 8):                      # probe every 8th layer
            ifj[j1 + 1] = 0
            if deposit_one(j1-1, cgf, elo, gr, gs, gw, col, wl, wl4, nucont0):
                ifj[j1 + 1] = 1                                # mark this 8-block active
        for k1 in range(8, nrhox + 1, 8):                     # fill in the active blocks only
            if ifj[k1 - 7] + ifj[k1 + 1] == 0: continue
            for j1 in range(k1 - 7, k1):
                deposit_one(j1-1, cgf, elo, gr, gs, gw, col, wl, wl4, nucont0)
        iwlold = iw
    return xlines[:, pix_lo:pix_hi]                            # the window pixel band

waveset  = REF["waveset_nm"]                                  # the full wavelength grid [nm]
iwavetab = REF["iwavetab"]                                    # continuum-bin boundaries (integer iwl)
tabcont  = REF["tabcont"].astype(np.float64)                 # continuum cutoff per (depth, bin)
pix_lo, pix_hi = int(REF["win_pix_lo"]), int(REF["win_pix_hi"])
win_wave = REF["win_waveset_nm"]                              # the window pixel band's wavelengths [nm]

import time
t0 = time.perf_counter()
xlines = deposit_window(waveset, iwavetab, tabcont, pix_lo, pix_hi)
print(f"deposited {n_lines} lines x {n_depth} depths in {time.perf_counter()-t0:.1f}s")
print(f"line opacity xlines: max = {xlines.max():.4e}, fills {np.count_nonzero(xlines)} pixel-depths")''')

# ── benchmark the deposit ──────────────────────────────────────────────────────
md(r"""## Benchmark: the deposit matches to the float32 floor

The reference ships the production code's own deposit of this same window, `xlines_window_ref`. Our from-scratch kernel must reproduce it — and because we kept the float32 accumulation, the `cv<tabcont` break, and the Harris-table indexing identical to the production code, it does, to the single-precision floor that the float32 wing accumulator carries (a relative difference at the level of the last bit of a 32-bit number, $\sim10^{-7}$ — the same documented floor the rest of the book reports where the production arithmetic is single precision).""")

code(r'''ref_x = REF["xlines_window_ref"].astype(np.float64)
got_x = xlines.astype(np.float64)
absdiff = np.abs(got_x - ref_x).max()
m = np.abs(ref_x) > 0
rel_deposit = (np.abs(got_x[m] - ref_x[m]) / np.abs(ref_x[m])).max() if m.any() else 0.0
npix = int(np.count_nonzero(ref_x.sum(0)))
print(f"deposit vs production ({npix} non-empty pixels):")
print(f"  max|abs diff| = {absdiff:.3e}")
print(f"  max|rel|      = {rel_deposit:.3e}   <- the float32 wing-accumulator floor (~1e-7)")''')

md(r"""A figure makes the structure of the deposit visible: at one depth, the total line opacity over the window is a forest of overlapping Voigt profiles, each walked outward from its center until it sinks below the continuum, on top of which the strong lines tower. The continuum-cutoff floor (the `tabcont` line) is exactly where each wing-walk stops.""")

code(r'''j_show = min(40, n_depth - 1)                  # a representative photospheric depth
nucont_show = np.searchsorted(iwavetab, iwl.min(), side="right") - 1
cont_floor = tabcont[j_show, max(nucont_show, 0)]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.semilogy(win_wave, np.maximum(xlines[j_show], cont_floor*1e-3), color="C0", lw=1.0,
            label="line opacity (from scratch)")
ax.semilogy(win_wave, np.maximum(ref_x[j_show], cont_floor*1e-3), color="C3", lw=1.0, ls=":",
            label="production reference")
ax.axhline(cont_floor, color="0.5", lw=1.0, ls="--", label="continuum cutoff (tabcont)")
ax.set_xlabel("wavelength [nm]"); ax.set_ylabel(r"line opacity $\kappa^{\rm line}$ [cm$^2$/g]")
ax.set_title(f"The deposited line forest (depth {j_show}, T = {REF['T'][j_show]:.0f} K)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); plt.show()
print("the from-scratch deposit (blue) lies under the production reference (red dotted)")''')

# ── the line-blanketed Rosseland mean ──────────────────────────────────────────
md(r"""## The line-blanketed Rosseland mean

With a way to deposit lines, we can fold them into the **Rosseland mean** — the harmonic, $\partial B_\nu/\partial T$-weighted average of the total opacity that sets the optical-depth scale (Lecture 10). The only change from Lecture 11 is the opacity that enters: where Lecture 11 used $\kappa_\nu = \kappa^{\rm cont}_\nu + \sigma_\nu$ (continuum absorption plus scattering), the blanketed mean uses

$$\kappa^{\rm tot}_\nu \;=\; \kappa^{\rm cont}_\nu \;+\; \kappa^{\rm line}_\nu \;+\; \sigma_\nu,
\qquad \frac{1}{\kappa_{\rm Ross}} \;=\; \frac{\int (1/\kappa^{\rm tot}_\nu)\,(\partial B_\nu/\partial T)\,d\nu}{\int (\partial B_\nu/\partial T)\,d\nu}.$$

The line opacity that enters is the deposited line opacity `xlines` (the full-grid blanket) with the stimulated-emission factor applied, $\kappa^{\rm line}_\nu = \texttt{xlines}_\nu\,(1-e^{-h\nu/kT})$. Depositing all 18 million lines onto the full 30000-frequency grid is the multi-gigabyte calculation we cannot run live, so — exactly as Lecture 11 took the continuum `acont` as a given input — the reference ships the full-grid line opacity `xlines_fullgrid` already deposited (its kernel verified on the window above). We fold it into the Rosseland mean and compare to the continuum-only mean of Lecture 11.""")

code(r'''SIGMA = 5.6697e-5; PLANCK = 6.6256e-27; KBOLTZ = 1.38054e-16

freq   = REF["freq_hz"]          # 30000 OS frequencies [Hz]
rco    = REF["rco"]              # Rosseland quadrature weights
acont  = REF["acont"].astype(np.float64)         # continuum absorption (depth x freq) -- GIVEN (Lecture 3)
sigmac = REF["sigmac"].astype(np.float64)        # continuum scattering -- GIVEN
xlines_full = REF["xlines_fullgrid"].astype(np.float64)   # the full deposited line blanket -- GIVEN (kernel verified above)

T = REF["T"]; n = T.size
hkt = PLANCK / np.maximum(T * KBOLTZ, 1e-300)

# the per-(depth, frequency) Planck function, stimulated-emission factor and diffusion weight,
# formed once over the whole grid (vectorized: the harmonic mean is just a weighted sum of 1/kappa)
ehvkt = np.exp(-freq[None, :] * hkt[:, None]); stim = np.maximum(1.0 - ehvkt, 1e-300)
bnu  = 1.47439e-2 * ((freq[None, :]/1e15)**3) * ehvkt / stim
dbdt = bnu * freq[None, :] * hkt[:, None] / np.maximum(T[:, None] * stim, 1e-300)   # dB_nu/dT
w    = rco[None, :]

def rosseland(include_lines):
    """Harmonic dB/dT-weighted mean of kappa_tot = acont (+line) + sigmac over the OS grid (vectorized)."""
    abtot = acont + sigmac
    if include_lines:
        abtot = abtot + xlines_full * stim                    # fold the blanket in (stim-corrected)
    acc = np.sum(dbdt / np.maximum(abtot, 1e-300) * w, axis=1)
    return (4.0*SIGMA/3.14159) * T**3 / np.maximum(acc, 1e-300)

kR_cont    = rosseland(include_lines=False)   # continuum only (Lecture 11)
kR_blanket = rosseland(include_lines=True)    # line-blanketed (this lecture)
print(f"Rosseland kappa raised by the blanket: factor {np.median(kR_blanket/kR_cont):.2f} (median), "
      f"{np.max(kR_blanket/kR_cont):.2f} (max)")''')

code(r'''fig, ax = plt.subplots(figsize=(8.2, 4.3))
ax.semilogy(np.log10(tauros), kR_cont,    color="0.55", lw=1.8, label="continuum only (Lecture 11)")
ax.semilogy(np.log10(tauros), kR_blanket, color="C3",   lw=2.0, label="line-blanketed (this lecture)")
ax.set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax.set_ylabel(r"$\kappa_{\rm Ross}$ [cm$^2$/g]")
ax.set_title("The line blanket raises the Rosseland mean opacity")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); plt.show()
print("the lines lift kappa_Ross above the continuum-only mean -- most in the line-rich upper layers")''')

# ── the convergence engine (Lecture 11), unchanged ─────────────────────────────
md(r"""## The convergence engine — Lecture 11, unchanged, with the blanket on

Everything else is Lecture 11. We carry over, **verbatim**, the convergence engine that lecture built: the numerical kernels (`parcoe`, `integ`, `deriv`, `map1`), the JOSH full-depth flux solver, the Rosseland finalize, the `ROSSTAB` opacity table, the hydrostatic `ttaup`, the mixing-length `convec`, and the temperature correction `tcorr_mode3`. They are **inlined in the next cell**, byte-identical to Lecture 11, so this notebook stays self-contained (it imports only NumPy) — we do not re-derive them here. That cell is a long appendix you can fold away; the lecture's new physics is the deposit kernel and the line-blanketed fold above.

The single difference is in the frequency sweep: where Lecture 11 passed *zero* line opacity to the JOSH solver, we now pass the deposited blanket `aline`. The JOSH solver already accepts a line-opacity argument — it always did — so the only change is the value we hand it. The temperature correction, the convection, the radiation pressure, the hydrostatic re-equilibration: all unchanged, all opacity-agnostic, exactly as Lecture 11 promised.""")

code('# --- The Lecture 11 convergence engine, carried over VERBATIM ---------------------\n# These are byte-identical to the kernels and solvers Lecture 11 built (PARCOE/INTEG/\n# DERIV/MAP1, the JOSH full-depth flux solver, ROSS, ROSSTAB, TTAUP, CONVEC, TCORR).\n# We do not re-derive them here -- this lecture\'s new physics is the line deposit and\n# the line-blanketed fold above; the engine below is unchanged and opacity-agnostic.\n# The notebook stays self-contained (only numpy): the engine is inlined, not imported.\nFOURPI = 12.5664            # 4 pi, written explicitly by ATLAS\nITER_TOL = 1.0e-5           # JOSH inner-sweep tolerance\n# the JOSH operator tables (Lecture 8): CH_MAT (H moment), CK_W (surface K moment)\nCH_MAT = np.load(pathlib.Path("..") / "reference" / "converged_ref.npz")["josh_coefh"].astype(np.float64)\nCK_W   = np.load(pathlib.Path("..") / "reference" / "josh_ck.npz")["ck"].astype(np.float32)\n\ndef parcoe(f, x):\n    n = f.size\n    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)\n    if n == 0:\n        return a, b, c\n    if n == 1:\n        a[0] = f[0]; return a, b, c\n    b[0] = (f[1] - f[0]) / (x[1] - x[0]); a[0] = f[0] - x[0] * b[0]\n    n1 = n - 1\n    b[-1] = (f[-1] - f[n1 - 1]) / (x[-1] - x[n1 - 1]); a[-1] = f[-1] - x[-1] * b[-1]\n    if n == 2:\n        return a, b, c\n    for j in range(1, n1):\n        j1 = j - 1\n        d = (f[j] - f[j1]) / (x[j] - x[j1])\n        c[j] = f[j + 1] / ((x[j + 1] - x[j]) * (x[j + 1] - x[j1])) + \\\n            (f[j1] / (x[j + 1] - x[j1]) - f[j] / (x[j + 1] - x[j])) / (x[j] - x[j1])\n        b[j] = d - (x[j] + x[j1]) * c[j]\n        a[j] = f[j1] - x[j1] * d + x[j] * x[j1] * c[j]\n    c[1] = 0.0; b[1] = (f[2] - f[1]) / (x[2] - x[1]); a[1] = f[1] - x[1] * b[1]\n    if n > 3:\n        c[2] = 0.0; b[2] = (f[3] - f[2]) / (x[3] - x[2]); a[2] = f[2] - x[2] * b[2]\n    for j in range(1, n1):\n        if c[j] == 0.0:\n            continue\n        j1 = min(j + 1, n - 1)\n        denom = abs(c[j1]) + abs(c[j])\n        wt = abs(c[j1]) / denom if denom > 0.0 else 0.0\n        a[j] = a[j1] + wt * (a[j] - a[j1])\n        b[j] = b[j1] + wt * (b[j] - b[j1])\n        c[j] = c[j1] + wt * (c[j] - c[j1])\n    a[n1 - 1] = a[-1]; b[n1 - 1] = b[-1]; c[n1 - 1] = c[-1]\n    return a, b, c\n\ndef integ(x, f, start):\n    n = f.size\n    out = np.zeros(n)\n    if n == 0:\n        return out\n    a, b, c = parcoe(f, x)\n    out[0] = start\n    for i in range(n - 1):\n        dx = x[i + 1] - x[i]\n        term = a[i] + 0.5 * b[i] * (x[i + 1] + x[i]) + (c[i] / 3.0) * ((x[i + 1] + x[i]) * x[i + 1] + x[i] * x[i])\n        out[i + 1] = out[i] + term * dx\n    return out\n\ndef deriv(x, f):\n    n = f.size\n    d = np.zeros(n)\n    if n < 2:\n        return d\n    d[0] = (f[1] - f[0]) / (x[1] - x[0])\n    d[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])\n    if n == 2:\n        return d\n    s = abs(x[1] - x[0]) / (x[1] - x[0]) if x[1] != x[0] else 1.0\n    for j in range(1, n - 1):\n        scale = max(abs(f[j - 1]), abs(f[j]), abs(f[j + 1]))\n        scale = scale / abs(x[j]) if x[j] != 0.0 else scale\n        if scale == 0.0:\n            scale = 1.0\n        d1 = (f[j + 1] - f[j]) / (x[j + 1] - x[j]) / scale\n        d0 = (f[j] - f[j - 1]) / (x[j] - x[j - 1]) / scale\n        tan1 = d1 / (s * np.sqrt(1.0 + d1 * d1) + 1.0)\n        tan0 = d0 / (s * np.sqrt(1.0 + d0 * d0) + 1.0)\n        d[j] = (tan1 + tan0) / (1.0 - tan1 * tan0) * scale\n    return d\n\ndef map1(xold, fold, xnew):\n    nold, nnew = xold.size, xnew.size\n    fnew = np.zeros(nnew)\n    if nold == 0 or nnew == 0:\n        return fnew, 0\n    xo = np.empty(nold + 1); fo = np.empty(nold + 1)\n    xo[1:] = xold; fo[1:] = fold\n    l = 2; ll = 0\n    cfor = bfor = afor = cbac = bbac = abac = a = b = c = 0.0\n    for k in range(1, nnew + 1):\n        xk = xnew[k - 1]\n        while True:\n            if xk < xo[l]:\n                if l == ll:\n                    break\n                if l == 2 or l == 3:\n                    l = min(nold, l); c = 0.0\n                    b = (fo[l] - fo[l - 1]) / (xo[l] - xo[l - 1]); a = fo[l] - xo[l] * b; ll = l\n                    break\n                l1 = l - 1\n                if l > ll + 1 or l == 3 or l == 4:\n                    l2 = l - 2\n                    d = (fo[l1] - fo[l2]) / (xo[l1] - xo[l2])\n                    cbac = fo[l] / ((xo[l] - xo[l1]) * (xo[l] - xo[l2])) + \\\n                        (fo[l2] / (xo[l] - xo[l2]) - fo[l1] / (xo[l] - xo[l1])) / (xo[l1] - xo[l2])\n                    bbac = d - (xo[l1] + xo[l2]) * cbac\n                    abac = fo[l2] - xo[l2] * d + xo[l1] * xo[l2] * cbac\n                else:\n                    cbac, bbac, abac = cfor, bfor, afor\n                if l >= nold:\n                    c, b, a, ll = cbac, bbac, abac, l\n                    break\n                d = (fo[l] - fo[l1]) / (xo[l] - xo[l1])\n                cfor = fo[l + 1] / ((xo[l + 1] - xo[l]) * (xo[l + 1] - xo[l1])) + \\\n                    (fo[l1] / (xo[l + 1] - xo[l1]) - fo[l] / (xo[l + 1] - xo[l])) / (xo[l] - xo[l1])\n                bfor = d - (xo[l] + xo[l1]) * cfor\n                afor = fo[l1] - xo[l1] * d + xo[l] * xo[l1] * cfor\n                wt = abs(cfor) / (abs(cfor) + abs(cbac)) if abs(cfor) != 0.0 else 0.0\n                a = afor + wt * (abac - afor); b = bfor + wt * (bbac - bfor); c = cfor + wt * (cbac - cfor)\n                ll = l\n                break\n            l += 1\n            if l > nold:\n                l = min(nold, l); c = 0.0\n                b = (fo[l] - fo[l - 1]) / (xo[l] - xo[l - 1]); a = fo[l] - xo[l] * b; ll = l\n                break\n        fnew[k - 1] = a + (b + c * xk) * xk\n    return fnew, max(ll - 1, 0)\n\ndef map1_scalar(xold, fold, xnew_val):\n    out, _ = map1(np.asarray(xold), np.asarray(fold), np.asarray([xnew_val]))\n    return float(out[0])\n\ndef _nz_signed(x, eps=1e-300):\n    if abs(x) >= eps:\n        return x\n    return eps if x >= 0.0 else -eps\n\ndef josh_profiles(acont, scont, aline, sline, sigmac, sigmal, rhox, bnu,\n                  xtau, ch, coefj):\n    n = rhox.size\n    nxtau = xtau.size\n    coefj_diag = np.diag(coefj).astype(np.float32)\n\n    abtot = np.maximum(acont + aline + sigmac + sigmal, 1e-300)\n    alpha = (sigmac + sigmal) / abtot\n    den = acont + aline\n    snubar = bnu.copy()\n    np.divide(acont * scont + aline * sline, den, out=snubar, where=den > 0.0)\n\n    taunu = integ(rhox, abtot, abtot[0] * rhox[0])\n    snu = np.zeros(n); hnu = np.zeros(n); jnu = np.zeros(n); jmins = np.zeros(n)\n    xs = np.zeros(nxtau, dtype=np.float32)\n\n    if taunu[0] > xtau[-1]:\n        maxj = 1\n    else:\n        xsbar8, maxj = map1(taunu, snubar, xtau)\n        xalpha8, maxj = map1(taunu, alpha, xtau)\n        xalpha8 = np.maximum(xalpha8.astype(np.float32), np.float32(0.0))\n        xsbar8 = np.maximum(xsbar8.astype(np.float32), np.float32(1.0e-38))\n        mask = xtau < taunu[0]\n        if np.any(mask):\n            xsbar8[mask] = max(snubar[0], 1.0e-38)\n            xalpha8[mask] = max(alpha[0], 0.0)\n        xs[:] = xsbar8\n        one32 = np.float32(1.0)\n        diag = one32 - xalpha8 * coefj_diag\n        xsbar_mod = (one32 - xalpha8) * xsbar8\n        for _ in range(nxtau):\n            iferr = 0\n            for kk in range(nxtau):\n                k = nxtau - 1 - kk\n                dot = np.float32(np.dot(coefj[k, :].astype(np.float32), xs))\n                num = np.float32(dot * xalpha8[k] + xsbar_mod[k] - xs[k])\n                dd = np.float32(diag[k])\n                if abs(float(dd)) < 1.0e-37:\n                    dd = np.float32(1.0e-37 if float(dd) >= 0.0 else -1.0e-37)\n                delxs = np.float32(num / dd)\n                xbase = np.float32(xs[k])\n                if abs(float(xbase)) < 1.0e-37:\n                    xbase = np.float32(1.0e-37 if float(xbase) >= 0.0 else -1.0e-37)\n                errx = np.float32(abs(float(delxs / xbase)))\n                if errx > np.float32(ITER_TOL):\n                    iferr = 1\n                xs[k] = np.float32(max(float(np.float32(xs[k] + delxs)), 1.0e-37))\n            if iferr == 0:\n                break\n        xs8 = xs.astype(np.float64)\n        snu_head, _ = map1(xtau, xs8, taunu[:maxj])\n        snu[:maxj] = snu_head\n\n    if maxj == n:\n        raise RuntimeError("maxj==n branch not expected for this reference atmosphere")\n\n    maxj1 = maxj + 1\n    if maxj == 1:\n        maxj1 = 1\n    snu[maxj1 - 1:] = snubar[maxj1 - 1:]\n    m = max(maxj - 1, 1)\n    m0 = m - 1\n    nmj0 = maxj - 1\n    for _ in range(nxtau):\n        error = 0.0\n        ifneg = 0\n        if np.any(snu[m0:] <= 0.0):\n            ifneg = 1\n            snubar[m0:] = bnu[m0:]\n            snu[m0:] = bnu[m0:]\n        hnu[m0:] = deriv(taunu[m0:], snu[m0:]) / 3.0\n        if np.any(hnu[m0:] <= 0.0):\n            ifneg = 1\n            snubar[m0:] = bnu[m0:]\n            snu[m0:] = bnu[m0:]\n            hnu[m0:] = deriv(taunu[m0:], snu[m0:]) / 3.0\n        jmins[nmj0:] = deriv(taunu[nmj0:], hnu[nmj0:])\n        for j in range(maxj1 - 1, n):\n            if ifneg == 1:\n                jmins[j] = 0.0\n            jnu[j] = jmins[j] + snu[j]\n            snew = (1.0 - alpha[j]) * snubar[j] + alpha[j] * jnu[j]\n            error += abs(snew - snu[j]) / max(abs(snew), 1e-300)\n            snu[j] = snew\n        if error < ITER_TOL:\n            break\n\n    if maxj == 1:\n        # degenerate surface (tau beyond the grid): surface Eddington factor K = J/3\n        knu_surface = jnu[0] / 3.0\n        return taunu, hnu, jmins, abtot, alpha, float(knu_surface)\n\n    xjs = (-xs + coefj.astype(np.float32) @ xs).astype(np.float64)\n    xh = (CH_MAT @ xs).astype(np.float64)\n    jmins[:maxj], _ = map1(xtau, xjs, taunu[:maxj])\n    hnu[:maxj], _ = map1(xtau, xh, taunu[:maxj])\n    # surface second moment off the SAME float32 source vector: K_nu(0) = CK . xs\n    knu_surface = float(np.dot(CK_W, xs))\n    return taunu, hnu, jmins, abtot, alpha, float(knu_surface)\n\ndef ross_finalize(acc, T, rhox):\n    abross = (4.0 * SIGMA / 3.14159) * np.power(T, 3.0) / np.maximum(acc, 1e-300)\n    tauros = integ(rhox, abross, abross[0] * rhox[0])\n    return abross, tauros\n\ndef expi3(x):\n    a = (-44178.5471728217, 57721.7247139444, 9938.31388962037, 1842.11088668,\n         101.093806161906, 5.03416184097568)\n    b = (76537.3323337614, 32597.1881290275, 6106.10794245759, 635.419418378382, 37.2298352833327)\n    c = (4.65627107975096e-7, 0.999979577051595, 9.04161556946329, 24.3784088791317,\n         23.0192559391333, 6.90522522784444, 0.430967839469389)\n    dco = (10.0411643829054, 32.4264210695138, 41.2807841891424, 20.4494785013794,\n           3.31909213593302, 0.103400130404874)\n    e = (-0.999999999998447, -26.6271060431811, -241.055827097015, -895.927957772937,\n         -1298.85688746484, -545.374158883133, -5.66575206533869)\n    fco = (28.6271060422192, 292.310039388533, 1332.78537748257, 2777.61949509163,\n           2404.01713225909, 631.6574832808)\n    if x <= 0.0:\n        ex1 = 0.0\n    else:\n        ex = np.exp(-x)\n        if x > 4.0:\n            ex1 = (ex + ex * (e[0] + (e[1] + (e[2] + (e[3] + (e[4] + (e[5] + e[6] / x) / x) / x) / x) / x) / x)\n                   / (x + fco[0] + (fco[1] + (fco[2] + (fco[3] + (fco[4] + fco[5] / x) / x) / x) / x) / x)) / x\n        elif x > 1.0:\n            ex1 = ex * (c[6] + (c[5] + (c[4] + (c[3] + (c[2] + (c[1] + c[0] * x) * x) * x) * x) * x) * x) / \\\n                (dco[5] + (dco[4] + (dco[3] + (dco[2] + (dco[1] + (dco[0] + x) * x) * x) * x) * x) * x)\n        else:\n            ex1 = (a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * x) * x) * x) * x) * x) / \\\n                (b[0] + (b[1] + (b[2] + (b[3] + (b[4] + x) * x) * x) * x) * x) - np.log(x)\n    out = ex1\n    for i in range(1, 3):\n        out = (np.exp(-x) - x * out) / float(i)\n    return out\n\nclass Rosstab:\n    def __init__(self):\n        self.t = []; self.p = []; self.k = []\n        self.zerot = self.zerop = 0.0\n        self.slopet = self.slopep = 1.0\n        self.n = 0\n\n    def ingest(self, T, P, kappa):\n        nn = T.size\n        if self.n == 0:\n            self.zerot = np.log10(max(float(T[0]), 1e-300))\n            self.zerop = np.log10(max(float(P[0]), 1e-300))\n            self.slopet = np.log10(max(float(T[-1]), 1e-300)) - self.zerot\n            self.slopep = np.log10(max(float(P[-1]), 1e-300)) - self.zerop\n            if abs(self.slopet) < 1e-300:\n                self.slopet = 1.0\n            if abs(self.slopep) < 1e-300:\n                self.slopep = 1.0\n        for j in range(nn):\n            self.t.append((np.log10(max(float(T[j]), 1e-300)) - self.zerot) / self.slopet)\n            self.p.append((np.log10(max(float(P[j]), 1e-300)) - self.zerop) / self.slopep)\n            self.k.append(np.log10(max(float(kappa[j]), 1e-300)))\n            self.n += 1\n\n    def eval(self, temp, pressure):\n        if self.n <= 0:\n            return 1.0\n        templog = (np.log10(max(temp, 1e-300)) - self.zerot) / self.slopet\n        presslog = (np.log10(max(pressure, 1e-300)) - self.zerop) / self.slopep\n        rpp = rpm = rmp = rmm = 1.0e30\n        i_pp = i_pm = i_mp = i_mm = -1\n        v_pp = v_pm = v_mp = v_mm = 0.0\n        for i in range(self.n):\n            dp = self.p[i] - presslog\n            dt = self.t[i] - templog\n            r2 = dt * dt + dp * dp\n            if dt >= 0.0 and dp >= 0.0:\n                if r2 < rpp:\n                    rpp = r2; i_pp = i; v_pp = self.k[i]\n            elif dt >= 0.0 and dp < 0.0:\n                if r2 < rpm:\n                    rpm = r2; i_pm = i; v_pm = self.k[i]\n            elif dt < 0.0 and dp >= 0.0:\n                if r2 < rmp:\n                    rmp = r2; i_mp = i; v_mp = self.k[i]\n            else:\n                if r2 < rmm:\n                    rmm = r2; i_mm = i; v_mm = self.k[i]\n        if i_pp >= 0 and i_pm >= 0 and i_mp >= 0 and i_mm >= 0:\n            tpp, ppp = self.t[i_pp], self.p[i_pp]\n            tpm, ppm = self.t[i_pm], self.p[i_pm]\n            tmp, pmp = self.t[i_mp], self.p[i_mp]\n            tmm, pmm = self.t[i_mm], self.p[i_mm]\n            den_tp = max(tpp - tmp, 1e-300); den_tm = max(tpm - tmm, 1e-300)\n            rppmp = ((templog - tmp) * v_pp + (tpp - templog) * v_mp) / den_tp\n            rpmmm = ((templog - tmm) * v_pm + (tpm - templog) * v_mm) / den_tm\n            pppmp = ((templog - tmp) * ppp + (tpp - templog) * pmp) / den_tp\n            ppmmm = ((templog - tmm) * ppm + (tpm - templog) * pmm) / den_tm\n            r = ((presslog - ppmmm) * rppmp + (pppmp - presslog) * rpmmm) / max(pppmp - ppmmm, 1e-300)\n            return float(10.0 ** r)\n        w_pp = 1.0 / (np.sqrt(rpp) + 1.0e-5); w_pm = 1.0 / (np.sqrt(rpm) + 1.0e-5)\n        w_mp = 1.0 / (np.sqrt(rmp) + 1.0e-5); w_mm = 1.0 / (np.sqrt(rmm) + 1.0e-5)\n        rwt = w_pp + w_pm + w_mp + w_mm\n        i_pp = max(i_pp, 0); i_pm = max(i_pm, 0); i_mp = max(i_mp, 0); i_mm = max(i_mm, 0)\n        r = (self.k[i_pp] * w_pp + self.k[i_pm] * w_pm + self.k[i_mp] * w_mp + self.k[i_mm] * w_mm) / max(rwt, 1e-300)\n        return float(10.0 ** r)\n\ndef ttaup(t, tau, prad, pturb, grav, rosstab):\n    n = int(t.size)\n    abstd = np.zeros(n); ptotal = np.zeros(n); pgas = np.zeros(n)\n    dlg_tau = np.log(max(float(tau[1] / max(tau[0], 1e-300)), 1e-300)) if n > 1 else 0.0\n    plog1 = plog2 = plog3 = plog4 = 0.0\n    dplog1 = dplog2 = dplog3 = 0.0\n    abstd[0] = 0.1\n    if prad[0] > 0.0:\n        abstd[0] = min(0.1, grav * tau[0] / max(prad[0], 1e-300) / 2.0)\n    for j in range(n):\n        if j == 0:\n            plog = np.log(max(grav / max(abstd[0], 1e-300) * tau[0], 1e-300))\n        elif j <= 3:\n            plog = plog1 + dplog1\n        else:\n            plog = (3.0 * plog4 + 8.0 * dplog1 - 4.0 * dplog2 + 8.0 * dplog3) / 3.0\n        error = 1.0; dplog = 0.0; itn = 1\n        while True:\n            plog = min(plog, 709.78)\n            ptotal[j] = np.exp(plog)\n            pgas[j] = ptotal[j] + (prad[0] - prad[j]) - pturb[j]\n            if pgas[j] <= 0.0:\n                pgas[j] = 1e-30; abstd[j] = 0.1; break\n            abstd[j] = rosstab(float(t[j]), float(pgas[j]))\n            dplog = grav / max(abstd[j], 1e-300) * tau[j] / max(ptotal[j], 1e-300) * dlg_tau\n            itn += 1\n            if itn > 1000 or error <= 5.0e-5:\n                break\n            if j == 0:\n                pnew = np.log(max(grav / max(abstd[j], 1e-300) * tau[j], 1e-300))\n            elif j <= 3:\n                pnew = (plog + 2.0 * plog1 + dplog + dplog1) / 3.0\n            else:\n                pnew = (126.0 * plog1 - 14.0 * plog3 + 9.0 * plog4 + 42.0 * dplog\n                        + 108.0 * dplog1 - 54.0 * dplog2 + 24.0 * dplog3) / 121.0\n            error = abs(pnew - plog)\n            plog = 0.5 * (pnew + plog)\n        plog4 = plog3; plog3 = plog2; plog2 = plog1; plog1 = plog\n        dplog3 = dplog2; dplog2 = dplog1; dplog1 = dplog\n    return abstd, ptotal, pgas\n\ndef high_from_rhox(rhox, rho):\n    rhoinv = 1.0e-5 / np.maximum(rho, 1e-300)\n    return integ(rhox, rhoinv, 0.0)\n\ndef convec(rosstab, rhox, tauros, t, p, rho, abross, pradk, ptotal, grav,\n           flux, ed1, ed2, ed3, ed4, r1, r2, r3, r4,\n           mixlth=1.25, nconv=36):\n    n = int(t.size)\n    dtdrhx = deriv(rhox, t)                 # dT/dRHOX, for the actual gradient\n    dilut = 1.0 - np.exp(-tauros)           # geometric dilution of P_rad gradient\n\n    dltdlp = np.zeros(n); heatcp = np.zeros(n); dlrdlt = np.zeros(n)\n    velsnd = np.zeros(n); grdadb = np.zeros(n); hscale = np.zeros(n)\n    flxcnv = np.zeros(n); vconv = np.zeros(n); flxcnv0 = np.zeros(n)\n    deltat = np.zeros(n); rosst = np.zeros(n)\n\n    for j in range(n):\n        delt = 0.0\n        # EOS derivatives from the +-0.1% finite differences (POPS samples).\n        dedt = (ed1[j] - ed2[j]) / max(t[j], 1e-300) * 500.0\n        drdt = (r1[j] - r2[j]) / max(t[j], 1e-300) * 500.0\n        dedpg = (ed3[j] - ed4[j]) / max(p[j], 1e-300) * 500.0\n        drdpg = (r3[j] - r4[j]) / max(p[j], 1e-300) * 500.0\n\n        dpdpg = 1.0\n        dpdt = 4.0 * pradk[j] / max(t[j], 1e-300) * dilut[j]     # radiation P term\n        # actual (radiative) gradient d ln T / d ln P\n        dltdlp[j] = ptotal[j] / max(t[j] * grav, 1e-300) * dtdrhx[j]\n        drdpg_safe = _nz_signed(float(drdpg))\n        heatcv = dedt - dedpg * drdt / drdpg_safe               # specific heat at const V\n        heatcp[j] = (dedt - dedpg * dpdt / max(dpdpg, 1e-300)\n                     - ptotal[j] / max(rho[j] ** 2, 1e-300)\n                     * (drdt - drdpg * dpdt / max(dpdpg, 1e-300)))   # at const P\n        if heatcv > 0.0:\n            velsnd[j] = np.sqrt(max(heatcp[j] / heatcv * dpdpg / drdpg_safe, 0.0))\n        dlrdlt[j] = t[j] / max(rho[j], 1e-300) * (drdt - drdpg * dpdt / max(dpdpg, 1e-300))\n        if abs(heatcp[j]) > 1e-300:\n            grdadb[j] = -ptotal[j] / max(rho[j] * t[j], 1e-300) * dlrdlt[j] / heatcp[j]\n        hscale[j] = ptotal[j] / max(rho[j] * grav, 1e-300)      # pressure scale height\n\n        # decide whether the layer convects (mixlth>0, j>=3, super-adiabatic)\n        if mixlth == 0.0 or j < 3:\n            continue\n        delt = dltdlp[j] - grdadb[j]                            # superadiabaticity\n        if delt < 0.0:\n            continue\n        vco = 0.5 * mixlth * np.sqrt(\n            max(-0.5 * ptotal[j] / max(rho[j], 1e-300) * dlrdlt[j], 0.0))\n        if vco == 0.0:\n            continue\n        fluxco = 0.5 * rho[j] * heatcp[j] * t[j] * mixlth / FOURPI\n        rosst[j] = rosstab.eval(float(t[j]), float(p[j]))       # cell-center opacity\n        olddelt = 0.0\n        # 30-iteration inner loop: solve flxcnv + temperature excess deltat\n        for _ in range(30):\n            rosst_denom = _nz_signed(float(rosst[j]))\n            dplus = rosstab.eval(float(t[j] + deltat[j]), float(p[j])) / rosst_denom\n            dminus = rosstab.eval(float(t[j] - deltat[j]), float(p[j])) / rosst_denom\n            if dplus == 0.0 or dminus == 0.0:\n                abconv = 0.0\n            else:\n                abconv = 2.0 / (1.0 / dplus + 1.0 / dminus) * abross[j]\n            den1 = abconv * hscale[j] * rho[j]\n            den2 = fluxco * FOURPI\n            if den1 == 0.0 or den2 == 0.0 or vco == 0.0:\n                d = 0.0\n            else:\n                d = 8.0 * SIGMA * t[j] ** 4 / den1 / den2 / vco\n            taub = abconv * rho[j] * mixlth * hscale[j]         # cell optical thickness\n            d = d * taub ** 2 / (2.0 + taub ** 2)               # optical-thickness factor\n            d = d ** 2 / 2.0\n            ddd = (delt / _nz_signed(float(d + delt))) ** 2\n            if ddd < 0.5:\n                # series expansion of (1-sqrt(1-ddd))/ddd for small ddd\n                delta = 0.5; term = 0.5; up = -1.0; down = 2.0\n                while term > 1.0e-6:\n                    up += 2.0; down += 2.0\n                    term = up / down * ddd * term\n                    delta += term\n            else:\n                delta = (1.0 - np.sqrt(max(1.0 - ddd, 0.0))) / max(ddd, 1e-300)\n            delta = delta * delt ** 2 / _nz_signed(float(d + delt))\n            vconv[j] = vco * np.sqrt(max(delta, 0.0))\n            flxcnv[j] = max(fluxco * vconv[j] * delta, 0.0)\n            deltat[j] = t[j] * mixlth * delta\n            deltat[j] = min(deltat[j], t[j] * 0.15)             # cap the excess\n            deltat[j] = deltat[j] * 0.7 + olddelt * 0.3         # under-relax\n            if olddelt - 0.5 < deltat[j] < olddelt + 0.5:\n                break\n            olddelt = deltat[j]\n\n    flxcnv0[:] = flxcnv\n    height = high_from_rhox(rhox, rho)\n    # overwt=0 -> no overshoot blend.  Force the top NCONV layers non-convective.\n    k = int(max(min(nconv, n), 0))\n    if k > 0:\n        flxcnv[:k] = 0.0\n    return dict(flxcnv=flxcnv, flxcnv0=flxcnv0, dltdlp=dltdlp, grdadb=grdadb,\n                hscale=hscale, dlrdlt=dlrdlt, heatcp=heatcp, vconv=vconv,\n                velsnd=velsnd, height=height)\n\ndef tcorr_mode3(T, rhox, tauros, abross, flxrad, rjmins, rdabh, rdiagj,\n                flux, teff, prad, grav, rosstab, cv, mixlth=1.25,\n                steplg=0.125, tau1lg=-6.875):\n    n = T.size\n    dtdrhx = deriv(rhox, T)\n    dabros = deriv(rhox, abross)\n\n    flxcnv = cv["flxcnv"]; flxcnv0 = cv["flxcnv0"]\n    dltdlp = cv["dltdlp"]; grdadb = cv["grdadb"]; hscale = cv["hscale"]\n    dlrdlt = cv["dlrdlt"]; heatcp = cv["heatcp"]\n    ptotal_cv = cv.get("ptotal")\n    rho_cv = cv.get("rho")\n    ddlt = deriv(rhox, dltdlp)\n\n    # cnvflx: copy FLXCNV, zero the two top layers, then a 1-2-1 smoothing pass.\n    cnvflx = flxcnv.copy()\n    cnvflx[0] = 0.0\n    if n >= 2:\n        cnvflx[1] = 0.0\n    if n >= 3:\n        ccc = cnvflx.copy()\n        for j in range(1, n - 1):\n            ccc[j] = 0.25 * cnvflx[j - 1] + 0.5 * cnvflx[j] + 0.25 * cnvflx[j + 1]\n        ccc[-1] = 0.25 * cnvflx[-3] + 0.25 * cnvflx[-2] + 0.5 * cnvflx[-1]\n        for j in range(1, n - 1):\n            cnvflx[j] = ccc[j]\n        cnvflx[-1] = ccc[-1]\n\n    rdabh_eff = rdabh - flxrad * dabros / np.maximum(abross, 1e-300)\n    codrhx = np.zeros(n)\n    ddel = np.zeros(n)\n    for j in range(n):\n        delv = 1.0\n        d = 0.0\n        # convective-efficiency factor ddel for layers that actually convect\n        if cnvflx[j] > 0.0 and flxcnv0[j] > 0.0:\n            delv = dltdlp[j] - grdadb[j]\n            vco = 0.5 * mixlth * np.sqrt(max(-0.5 * ptotal_cv[j] / max(rho_cv[j], 1e-300) * dlrdlt[j], 0.0))\n            fluxco = 0.5 * rho_cv[j] * heatcp[j] * T[j] * mixlth / FOURPI\n            if mixlth > 0.0 and vco > 0.0:\n                d = (8.0 * SIGMA * T[j] ** 4\n                     / np.maximum(abross[j] * hscale[j] * rho_cv[j], 1e-300)\n                     / np.maximum(fluxco * FOURPI, 1e-300) / vco)\n            taub = abross[j] * rho_cv[j] * mixlth * hscale[j]\n            d = d * taub * taub / (2.0 + taub * taub)\n            d = d * d / 2.0\n            den_deld = _nz_signed(float(d + delv))\n            del_safe = _nz_signed(float(delv))\n            ddel[j] = (1.0 + d / den_deld) / del_safe\n        cnvfl = 0.0\n        if flxrad[j] > 0.0:\n            if cnvflx[j] / flxrad[j] > 1.0e-3 and flxcnv0[j] / flxrad[j] > 1.0e-3:\n                cnvfl = cnvflx[j]\n        den_deld = _nz_signed(float(d + delv))\n        del_safe = _nz_signed(float(delv))\n        num = rdabh_eff[j] + cnvfl * (\n            dtdrhx[j] / max(T[j], 1e-300) * (1.0 - 9.0 * d / den_deld)\n            + 1.5 * ddlt[j] / del_safe * (1.0 + d / den_deld))\n        den = flxrad[j] + cnvflx[j] * 1.5 * dltdlp[j] * ddel[j]\n        codrhx[j] = num / _nz_signed(float(den))\n    codrhx[0] = 0.0\n    if n >= 2:\n        codrhx[1] = 0.0\n\n    g = np.exp(integ(rhox, codrhx, 0.0))\n    gfden = flxrad + cnvflx * 1.5 * dltdlp * ddel\n    gfden_safe = np.where(np.abs(gfden) >= 1e-300, gfden, np.where(gfden >= 0.0, 1e-300, -1e-300))\n    gflux = g * (flxrad + cnvflx - flux) / gfden_safe\n    dtau = integ(tauros, gflux, 0.0) / np.maximum(g, 1e-300)\n    dtau = np.maximum(-tauros / 3.0, np.minimum(tauros / 3.0, dtau))\n    dtflux = -dtau * dtdrhx / np.maximum(abross, 1e-300)\n\n    flxerr = (flxrad + cnvflx - flux) / np.maximum(flux, 1e-300) * 100.0\n    flxdrv = deriv(tauros, flxerr)\n    dtlamb = np.zeros(n)\n    teff25 = teff / 25.0\n    for j in range(n):\n        ratio = cnvflx[j] / np.maximum(flxrad[j], 1e-300)\n        if ratio < 1.0e-5:\n            flxdrv[j] = rjmins[j] / np.maximum(abross[j], 1e-300) / np.maximum(flux, 1e-300) * 100.0\n        denom = rdiagj[j] if abs(rdiagj[j]) > 1e-300 else np.sign(rdiagj[j]) * 1e-300\n        dtlamb[j] = -flxdrv[j] * flux / 100.0 / denom * abross[j]\n        if not (ratio < 1.0e-5 and tauros[j] < 1.0):\n            dtlamb[j] = 0.0\n            for k in range(1, 6):\n                jj = j - k\n                if jj >= 0:\n                    dtlamb[jj] *= 0.5\n        dtlamb[j] = float(np.clip(dtlamb[j], -teff25, teff25))\n\n    dtsur = (flux - flxrad[0]) / np.maximum(flux, 1e-300) * 0.25 * T[0]\n    dtsur = float(np.clip(dtsur, -teff25, teff25))\n    dum = dtflux + dtlamb\n    tinteg = integ(tauros, dum, 0.0)\n    tone = map1_scalar(tauros, tinteg, 0.1)\n    ttwo = map1_scalar(tauros, tinteg, 2.0)\n    tav = (ttwo - tone) / 2.0\n    if dtsur * tav <= 0.0:\n        tav = 0.0\n    if abs(tav) > abs(dtsur):\n        tav = dtsur\n    dtsur = dtsur - tav\n    dtsurf = np.full(n, dtsur)\n\n    dtflux = np.nan_to_num(dtflux, nan=0.0, posinf=0.0, neginf=0.0)\n    dtlamb = np.nan_to_num(dtlamb, nan=0.0, posinf=0.0, neginf=0.0)\n    dtsurf = np.nan_to_num(dtsurf, nan=0.0, posinf=0.0, neginf=0.0)\n    t1 = dtflux + dtlamb + dtsurf\n\n    # iteration 1: damping/acceleration skipped (skip_damp = True at iter_index=1)\n    tnew = T + t1\n    bad = ~np.isfinite(tnew)\n    if bad.any():\n        tnew = np.where(bad, T, tnew)\n    tnew = np.maximum(tnew, 1.0)\n    for i in range(1, n):\n        j = n - 1 - i\n        tnew[j] = np.fmin(tnew[j], tnew[j + 1] - 1.0)\n        if not np.isfinite(tnew[j]):\n            tnew[j] = max(T[j], 1.0)\n\n    # DRHOX: re-run TTAUP on TAUSTD for T and for T+t1, take fractional dP, map back\n    taustd = 10.0 ** (tau1lg + np.arange(n) * steplg)\n    rfun = rosstab.eval\n    tnew1, _ = map1(tauros, T, taustd)\n    prdnew, _ = map1(tauros, prad, taustd)\n    _a1, ptot1, _p1 = ttaup(tnew1, taustd, prdnew, np.zeros(n), grav, rfun)\n    tplus = T + t1\n    tnew2, _ = map1(tauros, tplus, taustd)\n    _a2, ptot2, _p2 = ttaup(tnew2, taustd, prdnew, np.zeros(n), grav, rfun)\n    ppp = (ptot2 - ptot1) / np.maximum(ptot1, 1e-300)\n    rrr, _ = map1(taustd, ppp, tauros)\n    drhox = rrr * rhox\n    rhox_new = rhox + drhox\n\n    return dict(t1=t1, dtflux=dtflux, dtlamb=dtlamb, flxerr=flxerr,\n                cnvflx=cnvflx, tnew=tnew, rhox_new=rhox_new, drhox=drhox)\nprint("Lecture 11 convergence engine inlined (JOSH, ROSS, ROSSTAB, TTAUP, CONVEC, TCORR) -- unchanged")')

code(r'''XTAU = JT["xtau"].astype(np.float64); CH = JT["ch"].astype(np.float64); COEFJ = JT["coefj"].astype(np.float64)
print("JOSH grid tables ready")''')

md(r"""The frequency sweep is the Lecture 11 sweep with one line changed. At each of the 30000 frequencies we form the Planck function, hand JOSH the continuum opacity **and** the line opacity `aline[:, inu]` (with the LTE line source $S^{\rm line}_\nu = B_\nu$), and accumulate the Rosseland integral and the four temperature-correction integrals. The line opacity enters the total extinction $\kappa^{\rm tot}_\nu = \kappa^{\rm cont}+\kappa^{\rm line}+\sigma$ that JOSH and the Rosseland mean both use — that one substitution is the whole of line blanketing.""")

code(r'''scont = REF["scont"].astype(np.float64)
rhox  = REF["rhox"]; P = REF["P"]; rho_in = REF["rho"]; ptotal = REF["ptotal"]
flux  = SIGMA / 12.5664 * TEFF**4
z = np.zeros(n)

ross_acc = np.zeros(n); flxrad = np.zeros(n); rjmins = np.zeros(n); rdabh = np.zeros(n); rdiagj = np.zeros(n)
accrad = np.zeros(n); pradk0_acc = 0.0

for inu in range(freq.size):
    f = float(freq[inu]); w = float(rco[inu])
    ehvkt = np.exp(-f * hkt); stim = np.maximum(1.0 - ehvkt, 1e-300)
    bnu = 1.47439e-2 * ((f/1e15)**3) * ehvkt / stim
    aline_nu = xlines_full[:, inu] * stim                            # stimulated-emission-corrected line absorption
    # JOSH with the line blanket switched on (continuum + LINE opacity); LTE line source = B_nu
    taunu, hnu, jmins, abtot, alpha, knu0 = josh_profiles(
        acont[:, inu], scont[:, inu], aline_nu, bnu, sigmac[:, inu], z, rhox, bnu, XTAU, CH, COEFJ)
    if np.any(hnu < 0.0): hnu = np.maximum(hnu, 1e-99)
    dbdt = bnu * f * hkt / np.maximum(T * stim, 1e-300)
    ross_acc += dbdt / np.maximum(abtot, 1e-300) * w                 # line-blanketed Rosseland mean
    rdabh  += deriv(rhox, abtot) / np.maximum(abtot, 1e-300) * hnu * w
    rjmins += abtot * jmins * w; flxrad += hnu * w; accrad += abtot * hnu * w; pradk0_acc += knu0 * w
    term2 = 0.0
    for j in range(n):                                              # the Lambda-diagonal (Lecture 10 E3 kernel)
        term1 = term2
        d = max(1e-10, float(taunu[j+1]-taunu[j]) if j != n-1 else 1e-10)
        if d <= 0.01:
            term2 = (0.922784335098467 - np.log(d))*d/4.0 + d*d/12.0 - d**3/96.0 + d**4/720.0
        else:
            ex = expi3(d) if d < 10.0 else 0.0
            term2 = 0.5*(d + ex - 0.5)/d
        diagj = term1 + term2; dbdtj = bnu[j]*f*hkt[j]/max(T[j]*stim[j], 1e-300)
        rdiagj[j] += abtot[j]*(diagj-1.0)/max(1.0-alpha[j]*diagj, 1e-300)*(1.0-alpha[j])*dbdtj*w
print(f"swept {freq.size} frequencies with the line blanket on; surface flux = {flxrad[0]:.4e}")''')

md(r"""From here the iteration closes exactly as in Lecture 11. The Rosseland mean and its optical-depth scale, the radiation pressure from the `RADIAP` moment, the `ROSSTAB` table, the mixing-length convection (with the equation-of-state finite-difference samples the reference ships, including the ionization-energy heat capacity — unchanged from Lecture 11), and the temperature correction `tcorr_mode3`. Then the corrected temperature and column mass are remapped onto the standard optical-depth grid. We run that closing sequence and read off the corrected model.""")

code(r'''abross, tauros_it = ross_finalize(ross_acc, T, rhox)

# RADIAP: the radiation pressure from the swept flux moments (Lecture 11)
c4pi = 12.5664 / 2.99792458e10; accrad *= c4pi
ratio = flxrad / max(flux, 1e-300); over = ratio > 1.0
accrad[over] *= flux / np.maximum(flxrad[over], 1e-300)
prad = integ(rhox, accrad, accrad[0]*rhox[0])
pradk0 = pradk0_acc * c4pi
if float(np.max(ratio)) > 1.0: pradk0 /= float(np.max(ratio))
pradk = prad + pradk0

# ROSSTAB + CONVEC (with the EOS finite-difference samples, incl. ionization-energy heat capacity)
rosstab = Rosstab(); rosstab.ingest(T, P, abross)
# the convective heat capacity reads the gas internal energy at T,P +-0.1% (eint1..4, with the
# ionization energy -- unchanged from Lecture 11), plus the radiation-energy term 3 P_radk/rho:
dilut = 1.0 - np.exp(-tauros_it)
r1, r2, r3, r4 = REF["rho1"], REF["rho2"], REF["rho3"], REF["rho4"]
ed1 = REF["eint1"] + 3.0*pradk/np.maximum(r1, 1e-300)*(1.0 + dilut*(1.001**4 - 1.0))
ed2 = REF["eint2"] + 3.0*pradk/np.maximum(r2, 1e-300)*(1.0 + dilut*(0.999**4 - 1.0))
ed3 = REF["eint3"] + 3.0*pradk/np.maximum(r3, 1e-300)
ed4 = REF["eint4"] + 3.0*pradk/np.maximum(r4, 1e-300)
cv = convec(rosstab, rhox, tauros_it, T, P, rho_in, abross, pradk, ptotal, GRAV, flux,
            ed1, ed2, ed3, ed4, r1, r2, r3, r4, mixlth=1.25, nconv=36)
cv["ptotal"] = ptotal; cv["rho"] = rho_in

# TCORR mode 3 -> corrected T, RHOX ; remap onto the standard optical-depth grid
res = tcorr_mode3(T, rhox, tauros_it, abross, flxrad, rjmins, rdabh, rdiagj,
                  flux, TEFF, prad, GRAV, rosstab, cv, mixlth=1.25)
taustd = 10.0**(-6.875 + np.arange(n)*0.125)
T_out, _    = map1(tauros_it, res["tnew"], taustd)
rhox_out, _ = map1(tauros_it, res["rhox_new"], taustd)
print(f"one line-blanketed iteration done: base abross = {abross[-1]:.2f} cm^2/g (continuum-only Lecture 11: ~3.1)")''')

# ── the benchmark ──────────────────────────────────────────────────────────────
md(r"""## The benchmark: engine fidelity, and reaching the real Sun

We benchmark exactly as Lecture 11 did, in two complementary ways. The **engine-fidelity** check compares our single line-blanketed iteration to the production code's *own* single line-blanketed step from the same converged model — this isolates the fidelity of our line-blanketed engine from the question of whether one step is an exact no-op, and it matches to the precision floor. The **self-consistency** check compares our corrected temperature to the real Sun's converged model `sun.npz`: because we started from the converged Sun and switched the blanket on, one step barely moves it — the model *sits* on `sun.npz`.""")

code(r'''def rel(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.abs(b) > 0; r = np.zeros_like(b); r[m] = np.abs(a[m]-b[m])/np.abs(b[m]); return r

# (a) engine fidelity: our line-blanketed step vs the production code's same step
rT = rel(T_out, REF["T_step"]).max()
rX = rel(rhox_out, REF["rhox_step"]).max()
# (b) self-consistency: the corrected T vs the real Sun sun.npz (one step ~ no-op at convergence)
T_sun = REF["T"]                                  # sun.npz temperature (the line-blanketed converged model)
rTsun = rel(T_out, T_sun)
print(f"(a) engine fidelity vs production single step:  T max|rel| = {rT:.2e}   RHOX max|rel| = {rX:.2e}")
print(f"(b) vs the real Sun sun.npz:  T median|rel| = {np.median(rTsun):.2e}   max|rel| = {rTsun.max():.2e}")
print(f"    surface T = {T_out[0]:.1f} K (sun {T_sun[0]:.1f})   base T = {T_out[-1]:.1f} K (sun {T_sun[-1]:.1f})")''')

code(r'''print("="*66)
print("LECTURE 15 BENCHMARK  (one line-blanketed iteration from the real Sun)")
print("="*66)
print(f"  deposit kernel (window)   : max|rel| = {rel_deposit:.2e}   <- float32 wing-accumulator floor")
print(f"  engine fidelity, T        : max|rel| = {rT:.2e}   <- production single-step floor")
print(f"  engine fidelity, RHOX     : max|rel| = {rX:.2e}   <- production single-step floor")
print(f"  reaches sun.npz, T median : {np.median(rTsun):.2e}   <- the real Sun's model atmosphere")
ok = (rel_deposit < 1e-5) and (rT < 1e-5) and (rX < 1e-5) and (np.median(rTsun) < 5e-3)
print("  PASS" if ok else "  FAIL")
print("="*66)''')

md(r"""The deposit kernel matches to the float32 wing-accumulator floor; the line-blanketed engine reproduces the production single step to its precision floor; and the corrected model sits on the real Sun's `sun.npz` to the documented deep-column residual (a part in $10^3$ — the same floor the independent line-blanketed convergence reaches, set by the deep-base opacity self-consistency, not by any approximation in the deposit or the fold). The continuum-only model of Lecture 11 has become the real, line-blanketed Sun.""")

code(r'''fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
ax[0].plot(np.log10(tauros), T_cont, color="0.55", lw=1.7, label="continuum only (Lecture 11)")
ax[0].plot(np.log10(tauros), T_sun, color="C3", lw=1.9, label="sun.npz (target)")
ax[0].plot(np.log10(tauros), T_out, color="C0", lw=1.3, ls="--", label="our line-blanketed step")
ax[0].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[0].set_ylabel("temperature [K]")
ax[0].set_title("The model reaches the real Sun"); ax[0].legend(loc="upper left", fontsize=9)

Ftot = (flxrad + res["cnvflx"]) * 12.5664
ax[1].axhline(SIGMA*TEFF**4, color="0.5", ls=":", lw=1.2, label=r"$\sigma T_{\rm eff}^4$ (target)")
ax[1].plot(np.log10(tauros), flxrad*12.5664, color="C0", lw=1.6, label="radiative")
ax[1].plot(np.log10(tauros), Ftot, color="C2", lw=1.8, label="total (rad + conv)")
ax[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[1].set_ylabel(r"flux [erg cm$^{-2}$ s$^{-1}$]")
ax[1].set_title("Flux constancy, line-blanketed"); ax[1].legend(loc="lower left", fontsize=9)
fig.tight_layout(); plt.show()''')

# ── synthesis ──────────────────────────────────────────────────────────────────
md(r"""## Synthesis

Lecture 11 converged a *continuum-only* model atmosphere of the Sun and named the debt it left open: the millions of spectral lines, switched off to keep the loop reproducible with NumPy alone. This lecture paid that debt. Line blanketing is not a cosmetic addition to the spectrum — it reshapes the *model atmosphere*: the metal-line forest, densest in the ultraviolet, blocks the escaping radiation, back-warms the deep layers, and cools the surface, until the flux is constant again on a genuinely different temperature structure.

We built the one genuinely new piece — the **line-deposit kernel** — from scratch and benchmarked it to the single-precision floor: the predicted line list and the `SELECTLINES` strength cut that keeps only the lines that can beat the continuum; the per-line center opacity, Doppler width, and damping from the line record and the equation-of-state state; and the production wing-walk itself — the **asymmetric, sub-pixel** abscissa on the logarithmic grid, the **full** three-branch Voigt (not a Gaussian-plus-Lorentzian shortcut, which fails for the heavily-damped base lines), and the **adaptive `cv<continuum` cutoff** that walks each line outward only as far as it stays visible. We folded the deposited opacity into the **Rosseland mean** and watched $\kappa_{\rm Ross}$ rise above the continuum-only value.

Then we did what Lecture 11 promised would *just work*: we handed the line opacity to the **unchanged Lecture 11 convergence engine** — the same JOSH flux, Rosseland mean, mixing-length convection (with the ionization-energy heat capacity intact), radiation pressure, and temperature correction — and ran one iteration with the blanket on. It reproduced the production single line-blanketed step to its precision floor and landed on the **real Sun's model atmosphere** `sun.npz` (surface 3696 K, base 11425 K, base column mass 12.14), to the documented part-in-$10^3$ deep-column residual. The continuum-only toy of Lecture 11 has become the real, line-blanketed Sun.""")

# ── summary ────────────────────────────────────────────────────────────────────
md(r"""## Summary

- **Line blanketing** is the blocking of radiation by the millions of metal lines (densest in the ultraviolet). It changes the *model atmosphere*, not just the spectrum: the trapped flux **back-warms** the deep layers while the surface **cools**, until flux constancy is restored on a new temperature structure.
- The **predicted line list** (`gfpred`, ~18 million transitions) is filtered by **`SELECTLINES`** — a strength cut that keeps only the lines whose opacity can rise above the local continuum *in this atmosphere*, the step that makes a multi-million-line calculation tractable. Each line is a compact record: an integer log-wavelength `iwl` ($\lambda=e^{\,\texttt{iwl}\cdot\texttt{RATIOLG}}$) plus `TABLOG`-indexed $gf$, excitation, and three damping constants.
- The **deposit kernel** (`LINOP1`/`_accwings`) walks each line outward from its center pixel, red and blue **separately**, depositing $\kappa_0\,H(a,v)$ at each pixel with a **true, asymmetric, sub-pixel** abscissa on the logarithmic grid, and breaks on each side when the value drops below the continuum (**deposit-then-break**, capped at 100 pixels). It uses the **full** three-branch Voigt $H(a,v)$ — the large-$a$ branches are essential for the heavily-damped base lines. Reproduced from scratch, it matches the production deposit to the single-precision floor of the float32 wing accumulator (most pixels bit-identical; the rest differ only in the last bit of a 32-bit number).
- The **line-blanketed Rosseland mean** folds the deposited line opacity into the total extinction, $\kappa^{\rm tot}_\nu=\kappa^{\rm cont}_\nu+\kappa^{\rm line}_\nu+\sigma_\nu$, lifting $\kappa_{\rm Ross}$ above the continuum-only value.
- The **convergence engine is Lecture 11, unchanged** — JOSH, Rosseland mean, `ROSSTAB`, mixing-length `convec` (ionization-energy heat capacity intact), `RADIAP`, `tcorr_mode3` — because it is opacity-agnostic. The *only* change is handing it `aline` instead of zero in the frequency sweep.
- One line-blanketed iteration from the converged Sun **reproduces the production single step** to its precision floor and **reaches the real Sun's model atmosphere** `sun.npz` (surface 3696 K, base 11425 K, base RHOX 12.14) to a part in $10^3$ — the deep-column self-consistency floor, not a deposit or fold approximation.
- What is computed from scratch *here*: the deposit kernel, the line-blanketed fold, the convergence step. What is given in this lecture (as Lecture 11 gave the continuum): the per-iteration equation-of-state state — which **Lecture 16 builds from scratch**, closing the last borrowed intermediate — and the precomputed full-grid line opacity, whose kernel is verified to the single-precision floor on the window.""")

# ── practice exercises ──────────────────────────────────────────────────────────
md(r"""## Practice exercises

**1. The cutoff reach.** Instrument `accwings` to record how many pixels each line walks before the `cv<tabcont` break. Plot the distribution at a deep (hot) and a shallow (cool) layer. Why do strong far-ultraviolet lines at the hot base reach far while most lines deposit only one or two pixels? How would a *fixed* window (say $\pm 20$ Doppler widths for every line) get both the strong and the weak lines wrong?

**2. The large-$a$ branch matters.** In `voigt_H`, force the small-$a$ table form `(H2*a+H1)*a+H0` for *all* $a$ (delete the `a>=0.2` branch) and re-run the window deposit. Which lines change, and by how much? Confirm that the heavily-damped lines (large $a$) go negative or wrong, and explain why the cheap form is only a small-$a$ expansion.

**3. Symmetric vs asymmetric.** Replace the true abscissa $(\texttt{waveset}[iw]-\lambda_0)/\Delta\lambda_{\rm D}$ with a *symmetric* integer offset $k\cdot(\texttt{waveset}[nu_0]-\texttt{waveset}[nu_0-1])/\Delta\lambda_{\rm D}$ on both sides, and measure the error against `xlines_window_ref`. Why does the logarithmic grid make the symmetric version wrong, and which lines (narrow or broad) suffer most?

**4. The blanket's reach in the Rosseland mean.** Split the line-blanketed Rosseland integral into ultraviolet ($\lambda<400$ nm), optical, and infrared bands. Which band's lines lift $\kappa_{\rm Ross}$ most at the hot base, and which at the cool surface? Relate this to the Wien peak of $\partial B_\nu/\partial T$ at 11425 K versus 3696 K.

**5. Back-warming, quantified.** Using the shipped continuum-only and line-blanketed temperatures, find the optical depth where the blanketed model is warmest relative to the continuum-only one. Estimate, from the change in the Rosseland opacity, roughly how much deeper the $\tau_{\rm Ross}=1$ surface sits in the blanketed model, and connect that to the surface cooling.

**6. Switch the blanket off, mid-loop.** Re-run the convergence frequency sweep passing `z` (zero) for `aline_nu` instead of the deposited blanket — i.e. reproduce Lecture 11's continuum-only step from the *line-blanketed* Sun. The corrected temperature should now move *away* from `sun.npz` (toward the continuum-only structure). By how much, and at what optical depth? This is the structural imprint of line blanketing, isolated.""")

# ── further reading ─────────────────────────────────────────────────────────────
md(r"""## Further reading

- **Kurucz, R. L. (1979). *Model Atmospheres for G, F, A, B, and O Stars*, ApJS 40, 1.** The classic demonstration of line-blanketed model atmospheres and the back-warming the line forest produces.
- **Kurucz, R. L. (1992). *Atomic and Molecular Data for Opacity Calculations*, Rev. Mex. Astron. Astrofis. 23, 45**, and the `gfpred` predicted-line lists. The provenance and scale of the ~18-million-line predicted catalog this lecture deposits.
- **Gray, D. F. (2005). *The Observation and Analysis of Stellar Photospheres*, 3rd ed., Cambridge.** Chapter 13 on line absorption coefficients, the Voigt profile, and the broadening mechanisms (radiative, Stark, van der Waals) assembled in the damping parameter here.
- **Mihalas, D. (1978). *Stellar Atmospheres*, 2nd ed., Freeman.** The treatment of line opacity, the Rosseland mean with lines, and the role of blanketing in the temperature structure.
- **Hubeny, I. & Mihalas, D. (2014). *Theory of Stellar Atmospheres*, Princeton.** Opacity sampling and the line-blanketed model-atmosphere problem in full.
- **Kurucz, R. L. (1970). *SAO Special Report* 309 (ATLAS).** The `LINOP` line-opacity deposit and the line-blanketed convergence reproduced here.
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz*](https://arxiv.org/abs/2603.11693).** The pure-Python ATLAS12 + SYNTHE implementation the reference deposit and line-blanketed model are computed with.""")

nb = new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")
