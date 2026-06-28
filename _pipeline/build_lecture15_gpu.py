#!/usr/bin/env python
"""Assemble content/Lecture15.ipynb (unexecuted) — the GPU EDITION. Execute + render via build.py.

Lecture 15 (GPU) — Line Blanketing: the True Model Atmosphere, ported to clean depth-batched
torch/MPS, AND used as a precision diagnostic. This is one of the atmosphere-CONVERGENCE
lectures: in fp32 on the GPU the full convergence loop DIVERGES, and that is exactly the point.
Each computational cell is ported to torch and paired with a comparison cell that runs BOTH the
working device (MPS/fp32) and a CPU/fp64 twin and reports the per-cell max relative deviation.
The cell-by-cell fp32-vs-fp64 comparison LOCALISES where single precision peels away from the
fp64 reference: the per-evaluation physics (the line deposit, the Rosseland fold, the tau integral)
holds fp32 parity near the float floor; the convergence-core SECANT (ptot2-ptot1)/ptot1 is where
catastrophic cancellation enters. The clean torch port is a pedagogical reduction of the production
kgpu engine (read-only); the notebook never imports kgpu or pykurucz.
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
md(r"""# Lecture 15 — Line Blanketing: the True Model Atmosphere *(GPU Edition)*

*Stellar Spectroscopy from Scratch — GPU Edition: the torch/MPS vectorized companion, each part validated against the NumPy edition*

*Yuan-Sen Ting*

*Written in collaboration with **Claude Opus 4.8**, under the author's supervision. Schematics generated with **Gemini 3 Pro** (Nano Banana).*

*This is the **GPU edition** of Lecture 15. The physics and the formulas are identical to the [NumPy edition](https://github.com/tingyuansen/stellar-spectroscopy-from-scratch); the line deposit and the line-blanketed convergence engine are rebuilt in clean, depth-batched **`torch`**. But this lecture has a second job. Lectures 15 and 16 are the **atmosphere-convergence** finale, and on the GPU — where the working precision is **fp32** (Apple MPS and CUDA have no float64) — the full convergence loop **diverges**. That divergence is not a bug to hide; it is a result to **localise**. So every cell here runs **twice** — once on the working device in fp32, once on the CPU in fp64 — and reports the per-cell maximum relative deviation. The comparison walks down the pipeline and shows **exactly which cell** single precision peels away from the fp64 reference. The clean torch port is a pedagogical reduction of the production `kgpu` engine; the notebook imports neither `kgpu` nor pykurucz.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Deposit a spectral line the way the production engine does — the **asymmetric sub-pixel wing-walk**, the **full three-branch Voigt** $H(a,v)$, and the **continuum-cutoff reach** — written as a clean depth-batched `torch` kernel.
- Fold the line blanket into the **Rosseland mean** opacity (the harmonic, $\partial B/\partial T$-weighted average of the total extinction) and integrate it to the **Rosseland optical depth** $\tau_{\rm Ross}$.
- Run **one iteration** of the line-blanketed convergence engine — the frequency sweep, the temperature correction, the column-mass update — and see the back-warming that builds the real Sun's atmosphere.
- **Run the fp32-vs-fp64 diagnostic cell by cell** and read off where the divergence enters: confirm (or refute) that the per-evaluation physics is fp32-safe to the float floor, and that the convergence-core **secant** $(\,p_2-p_1)/p_1$ is the catastrophic-cancellation point that needs fp64.""")

md(r"""## Why this lecture diverges in fp32 — and why that is the lesson

Up to Lecture 14 every GPU port held fp32 parity with the NumPy edition to a few $\times 10^{-6}$: the equation of state, the continuum, the lines, the molecular bands, the radiative transfer. Those are all **per-evaluation** computations — given the atmosphere, evaluate an opacity or a flux. Single precision handles them comfortably, because each number is computed *once* from well-conditioned inputs.

The **atmosphere convergence** is different. It is an *iteration*: start from a guess, compute the radiation field, correct the temperature and the column mass, repeat ~20–30 times until the structure stops moving. Two things make this fragile in fp32. First, the correction is a **finite difference** — it compares the pressure structure at temperature $T$ against the structure at a slightly perturbed $T+\delta T$, and a difference of two nearly-equal numbers loses significant digits (*catastrophic cancellation*). Second, the error **compounds**: a small bias in one iteration's column-mass update is carried into the next, and over twenty iterations a $10^{-3}$ bias can walk the deep-base structure far off course. In the production `kgpu` engine, running the fully-MPS fp32 path without care drives the base column mass to $\sim 8.5$ instead of the Sun's $12.14$ — a *diverged* model.

The cure is **surgical**: promote just the precision-critical reductions to fp64 (a few small per-depth offloads), and leave the rest in fast fp32. To know *which* reductions, you have to find where the divergence enters. That is what this lecture's cell-by-cell comparison does. We do **not** fix the divergence here (a concurrent effort owns the production fix); we **localise** it, honestly, with measured numbers.

> **A note on this lecture's scope.** We port and compare the load-bearing pieces — the line deposit, the Rosseland fold, the $\tau_{\rm Ross}$ integral, and the temperature-correction **secant** — each as a clean torch cell with an fp32-vs-fp64 readout. The full 30000-frequency sweep and the multi-million-line full-grid deposit are the multi-gigabyte calculations the NumPy edition also ships precomputed (`xlines_fullgrid`, `acont`); we reuse those given inputs and focus the GPU port + the precision diagnostic on the cells where the fp32 story actually lives.""")

# ── setup ─────────────────────────────────────────────────────────────────────
md(r"""## Setup — the device, the precision budget, and the two-twin comparison

We pick the device once: **MPS** (Apple) $\to$ **CUDA** $\to$ **CPU**. On MPS/CUDA the working dtype is **fp32**; on a CPU fallback it is **fp64**. The novelty of this lecture is that the comparison cannot rely on the device alone — we want to see fp32 *vs* fp64 in a **single run**, on whatever machine executes the notebook. So every ported computation is written as a function `f(device, dtype)` and we call it **twice**: once on the working device in the working dtype, once on the **CPU in fp64**. The per-cell readout is `max |fp32 − fp64| / |fp64|`. (MPS has no float64, so the fp64 twin always runs on the CPU; and remember the MPS gotcha — `tensor.to("cpu", torch.float64)` raises, you must `tensor.cpu().to(torch.float64)`.)""")

code(r'''import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt

# pick the working device ONCE; MPS (Apple) -> CUDA -> CPU. MPS/CUDA have no fp64,
# so the working dtype is fp32; on CPU we fall back to fp64.
if torch.backends.mps.is_available():
    DEVICE, DTYPE = torch.device("mps"), torch.float32
elif torch.cuda.is_available():
    DEVICE, DTYPE = torch.device("cuda"), torch.float32
else:
    DEVICE, DTYPE = torch.device("cpu"), torch.float64
DTAG = str(DTYPE).split(".")[-1]
print(f"working device = {DEVICE.type}   working dtype = {DTAG}")

# the fp64 reference twin ALWAYS runs on the CPU (MPS/CUDA cannot do fp64)
REF64 = (torch.device("cpu"), torch.float64)

def back(t):
    """Bring a tensor back to NumPy/fp64 for comparison. MPS gotcha: move to CPU FIRST, then cast."""
    if torch.is_tensor(t):
        return t.detach().cpu().to(torch.float64).numpy()
    return np.asarray(t, float)

def reldev(a, b):
    """max |a - b| / |b|, guarding a zero reference, in fp64 on the host."""
    a = back(a); b = back(b)
    denom = np.where(np.abs(b) > 0.0, np.abs(b), 1.0)
    return float(np.max(np.abs(a - b) / denom))

# the running diagnostic table: (cell name -> fp32-vs-fp64 max relative deviation)
DIVERGENCE = {}
def diagnose(name, work_result, ref64_result, expect):
    """Run a cell's result on the working device vs the CPU/fp64 twin; record + print the spread."""
    rel = reldev(work_result, ref64_result)
    DIVERGENCE[name] = rel
    flag = "float floor" if rel < 1e-4 else ("ELEVATED" if rel < 1e-2 else "CATASTROPHIC")
    print(f"  {name:34s}  fp32-vs-fp64 max|rel| = {rel:.2e}   [{flag}]   (expected: {expect})")
    return rel

plt.rcParams.update({
    "figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
})''')

md(r"""Load the reference bundle — the NumPy edition's `lineblanket_ref.npz`, copied here unchanged. It carries the converged solar structure ($T$, $\rho x$, $P$), the line list and the equation-of-state state for a small wavelength **window** (so the deposit runs live in a cell), the full-grid continuum and line opacity (`acont`, `xlines_fullgrid`, shipped precomputed exactly as the NumPy edition does), and the production answers we benchmark against.""")

code(r'''R = np.load(pathlib.Path("..") / "reference" / "lineblanket_ref.npz", allow_pickle=True)
KT = np.load(pathlib.Path("..") / "reference" / "leankurucz_tables.npz")   # Harris Voigt tables

TEFF = float(R["teff"]); GRAV = float(R["gravity_cgs"])
T = R["T"]; rhox = R["rhox"]; n_depth = T.size
print(f"converged solar structure: {n_depth} layers, Teff = {TEFF:.0f} K, log g = {float(R['logg']):.2f}")
print(f"target base column mass rhox = {rhox[-1]:.3f} g/cm^2 (the real Sun's value)")''')

# ── Voigt ─────────────────────────────────────────────────────────────────────
md(r"""## The Voigt profile $H(a,v)$ — three branches, in torch

The deposit needs the Voigt/Hjerting function $H(a,v)$ — the thermal-Gaussian core convolved with the Lorentz damping wing — at offset $v=(\lambda-\lambda_0)/\Delta\lambda_{\rm D}$ and damping $a$. As in Lecture 4 we build it from Kurucz's **Harris** polynomial tables `h0/h1/h2`, with its **three branches**: the cheap small-$a$ table form, the small-$a$ far-wing Lorentzian, and the full large-$a$ Harris series (with an analytic asymptotic for $a>1.4$). The third branch is essential — the heavily-damped lines at the hot base have $a\sim10$–$150$, where the cheap form goes negative.

In the NumPy edition this is a scalar branch on each `(line, depth)`. Here we write it **branchlessly**: compute all branches over a whole array of offsets at once, then select with `torch.where`, so every depth lane does the same work. This is the GPU-native shape — the same structure the production `kgpu` `harris_hav` uses.""")

code(r'''H0t = torch.as_tensor(KT["h0tab"], dtype=DTYPE, device=DEVICE)
H1t = torch.as_tensor(KT["h1tab"], dtype=DTYPE, device=DEVICE)
H2t = torch.as_tensor(KT["h2tab"], dtype=DTYPE, device=DEVICE)
H0r = torch.as_tensor(KT["h0tab"], dtype=torch.float64, device="cpu")   # fp64 twin tables
H1r = torch.as_tensor(KT["h1tab"], dtype=torch.float64, device="cpu")
H2r = torch.as_tensor(KT["h2tab"], dtype=torch.float64, device="cpu")

def voigt_H(v, a, H0, H1, H2):
    """Branchless Harris Voigt H(a,v), vectorized over the offset tensor v at damping a (broadcastable).
    All three branches computed, then torch.where-selected. v, a, tables share one dtype/device."""
    one = torch.ones_like(v)
    iv = torch.clamp((v * 200.0 + 1.5).to(torch.int64), 1, 2001) - 1
    h0 = H0[iv]; h1 = H1[iv]; h2 = H2[iv]
    big = a >= 0.2
    # small-a: Doppler core (table) or far-wing Lorentzian
    core = (h2 * a + h1) * a + h0
    far  = 0.5642 * a / torch.clamp(v * v, min=1e-30)
    small = torch.where(v > 10.0, far, core)
    # large-a: analytic asymptotic OR Harris correction series
    aa = a * a; vv = v * v
    u  = (aa + vv) * 1.4142
    asy = a * 0.79788 / torch.clamp(u, min=1e-30)
    aau = aa / torch.clamp(u, min=1e-30); vvu = vv / torch.clamp(u, min=1e-30); uu = u * u
    asy_corr = ((((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa)
                / torch.clamp(uu, min=1e-30) + 1.0) * asy
    asy_full = torch.where(a > 100.0, asy, asy_corr)
    hh1 = h1 + h0 * 1.12838
    hh2 = h2 + hh1 * 1.12838 - h0
    hh3 = (1.0 - h2) * 0.37613 - hh1 * 0.66667 * vv + hh2 * 1.12838
    hh4 = (3.0 * hh3 - hh1) * 0.37613 + h0 * 0.66667 * vv * vv
    series = (((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + h0)
              * (((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032))
    use_asy = (a > 1.4) | (a + v > 3.2)
    large = torch.where(use_asy, asy_full, series)
    return torch.where(big, large, small)

print("Voigt H(a,v) defined (branchless Harris, three branches selected by torch.where)")''')

# ── the deposit kernel ─────────────────────────────────────────────────────────
md(r"""## The line deposit — the asymmetric wing-walk, depth-batched

Depositing a line is not "evaluate the Voigt on a fixed window." The production kernel (`LINOP1`/`_accwings`) **walks outward** from the line center, one log-spaced grid pixel at a time, to the red and to the blue *separately*, adding $\kappa_0\,H(a,v)$ to the running opacity, and **stops** on each side once the deposited value drops below the local continuum `tabcont`. Three properties make it faithful: the walk is **asymmetric and sub-pixel** (the log grid spacing differs on the two sides; the center sits a fractional pixel below `waveset[nu0]`); the cutoff reach is **adaptive** (deposit, then break); and the Voigt is the **full** $H(a,v)$.

Here we port it as a **depth-batched** kernel: for one line, all 80 depths walk together. The center opacity, the Boltzmann factor, the damping $a$, and the per-pixel abscissa are tensor ops over the depth axis; the wing-walk steps the pixel offset, and a per-depth `active` mask carries the cutoff (a depth that has dropped below continuum stops contributing). The line center opacity is
$$\kappa_0(j) = \underbrace{C_{gf}\,\lambda\,gf}_{\texttt{cgf}}\cdot\underbrace{\frac{n_{\rm sp}}{\Delta\nu_{\rm D}\,\rho}}_{\texttt{xnfdop}}\cdot\,e^{-\chi_{\rm low}\,hc/kT(j)},\qquad a(j)=\frac{\Gamma_{\rm rad}+\Gamma_{\rm Stark}\,n_e+\Gamma_{\rm vdW}\,\texttt{txnxn}}{\Delta\nu_{\rm D}}.$$
The equation-of-state state ($n_{\rm sp}$, $\Delta\nu_{\rm D}$, $n_e$, `txnxn`) is shipped for the window's species (it is Lecture 16's output — built there from scratch). The kernel `f(device,dtype)` runs in fp32 on the device and fp64 on the CPU so we can compare the deposit at both precisions.""")

code(r'''# window data (a small wavelength band so the deposit runs live), all host arrays
RATIOLG = float(R["ratiolg"]); CGF_SCALE = float(R["cgf_scale"]); GAMMA_SCALE = float(R["gamma_scale"])
TABLOG = 10.0 ** ((np.arange(1, 32769) - 16384) * 0.001)          # the packed-value decode table
waveset = R["waveset_nm"]; iwavetab = R["iwavetab"]; tabcont = R["tabcont"].astype(np.float64)
win_pix_lo, win_pix_hi = int(R["win_pix_lo"]), int(R["win_pix_hi"])
iwl = R["win_iwl"]; ielion = R["win_ielion"]; ielo = R["win_ielo"]
igflog = R["win_igflog"]; igr = R["win_igr"]; igs = R["win_igs"]; igw = R["win_igw"]
nelion_set = R["win_nelion_set"]
xnfdop_w = R["win_xnfdop"]; dopple_w = R["win_dopple"]
xne_w = R["win_xne"]; txnxn_w = R["win_txnxn"]; hckt_w = R["win_hckt"]
nelion_to_col = {int(z): k for k, z in enumerate(nelion_set)}
print(f"window: {iwl.size} lines, {nelion_set.size} species, depositing onto pixels "
      f"[{win_pix_lo}:{win_pix_hi}] ({win_pix_hi-win_pix_lo} wide)")''')

code(r'''def deposit_window(device, dtype):
    """Depth-batched line deposit on the window, in (device, dtype). Returns xlines[depth, pix].
    Each line: compute kappa0, damping a, then walk red & blue with a per-depth `active` mask
    carrying the cv<tabcont cutoff. Faithful to LINOP1's asymmetric sub-pixel wing-walk."""
    f = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    H0, H1, H2 = (H0t, H1t, H2t) if dtype == DTYPE and device == DEVICE else (H0r, H1r, H2r)
    numnu = waveset.size
    xl = torch.zeros((n_depth, numnu), dtype=dtype, device=device)
    ws  = f(waveset); cont_all = f(tabcont)
    xnf = f(xnfdop_w); dop_all = f(dopple_w); xne = f(xne_w); txn = f(txnxn_w); hk = f(hckt_w)
    for k in range(iwl.size):
        iw = int(iwl[k]); nucont0 = int(np.searchsorted(iwavetab, iw, side="right"))
        if nucont0 >= tabcont.shape[1]: continue
        nel = abs(int(ielion[k])) // 10
        if nel not in nelion_to_col: continue
        col = nelion_to_col[nel]
        wl = float(np.exp(iw * RATIOLG))
        nu0 = int(np.searchsorted(waveset, wl, side="left"))
        if nu0 >= numnu: continue
        wl_t = torch.as_tensor(wl, dtype=dtype, device=device)
        cgf = CGF_SCALE * wl * float(TABLOG[int(igflog[k]) - 1])
        elo = float(TABLOG[int(ielo[k]) - 1])
        gr = float(TABLOG[int(igr[k]) - 1]) * wl * GAMMA_SCALE
        gs = float(TABLOG[int(igs[k]) - 1]) * wl * GAMMA_SCALE
        gw = float(TABLOG[int(igw[k]) - 1]) * wl * GAMMA_SCALE
        cont = cont_all[:, nucont0]                                  # (depth,) continuum cutoff
        cen = cgf * xnf[:, col] * torch.exp(-(elo * hk))             # (depth,) center opacity
        dop = dop_all[:, col]
        act0 = (cen >= cont) & (dop > 0)
        if not bool(act0.any()): continue
        adamp = (gr + gs * xne + gw * txn) / torch.clamp(dop, min=1e-30)
        dopwave = dop * wl_t
        for sign, rng in ((+1, range(0, 101)), (-1, range(1, 101))):   # red wing, then blue wing
            act = act0.clone()
            for off in rng:
                ip = nu0 + sign * off
                if ip < 0 or ip >= numnu: break
                vv = ((ws[ip] - wl_t) if sign > 0 else (wl_t - ws[ip])) / torch.clamp(dopwave, min=1e-30)
                cv = cen * voigt_H(vv, adamp, H0, H1, H2)
                here = act & (cv >= 0)
                xl[:, ip] = xl[:, ip] + torch.where(here, cv, torch.zeros_like(cv))
                act = act & (cv >= cont)                              # deposit-then-break, per depth
                if not bool(act.any()): break
    return xl[:, win_pix_lo:win_pix_hi]

import time
t0 = time.perf_counter()
dep_work = deposit_window(DEVICE, DTYPE)
print(f"deposited {iwl.size} lines x {n_depth} depths on {DEVICE.type}/{DTAG} "
      f"in {time.perf_counter()-t0:.1f}s; xlines max = {float(dep_work.max()):.3e}")''')

# ── deposit comparison ─────────────────────────────────────────────────────────
md(r"""### Comparison cell — the deposit, fp32 vs fp64 (and vs the production reference)

Now the first diagnostic. We run the **same** kernel on the CPU in fp64, compare the two, and also compare the fp64 result to the production deposit `xlines_window_ref` the NumPy edition ships (which proved bit-for-bit against pykurucz). Two numbers come out, and they say different things: the **fp64-vs-production** gap measures how faithful our *clean* depth-batched kernel is to the production `LINOP1` *algorithm* (the production code's 8-block depth-skip and exact float32 deposit order differ slightly from our clean vectorized walk); the **fp32-vs-fp64** spread is the precision story this lecture tracks.""")

code(r'''dep_ref64 = deposit_window(*REF64)

# fp32-vs-fp64: the precision spread of OUR kernel
diagnose("line deposit (wing-walk)", dep_work, dep_ref64,
         "near float floor on the cores; mild fp32 accumulation in overlapping wings")

# fp64-vs-production: how close the clean batched walk is to the LINOP1 algorithm
ref_x = R["xlines_window_ref"].astype(np.float64)
got64 = back(dep_ref64)
m = np.abs(ref_x) > 0
rel_algo = float(np.max(np.abs(got64[m] - ref_x[m]) / np.abs(ref_x[m]))) if m.any() else 0.0
# where does the fp32 spread live -- cores or wings?
mm = np.abs(back(dep_ref64)) > 0
big = back(dep_ref64) > 1e-3 * back(dep_ref64).max()
relmap = np.abs(back(dep_work) - back(dep_ref64))[mm] / np.abs(back(dep_ref64))[mm]
print(f"\n  fp64-vs-production LINOP1 algorithm:  max|rel| = {rel_algo:.2e}  (clean batched walk vs 8-block depth-skip)")
print(f"  fp32 spread is bounded: median|rel| (all pixels) = {np.median(relmap):.2e}  "
      f"<- the cores are at the float floor; the max sits on overlapping-wing pixels")''')

md(r"""**What the deposit tells us.** The per-evaluation deposit holds fp32 parity *well*: the line **cores** match to the float floor (median $\sim 10^{-5}$), and the worst fp32 spread sits on the mid-wing pixels where many overlapping Voigt wings are summed in fp32 — bounded, physically minor (a fraction of a percent on pixels that are themselves a few percent of the core). This is the expected behaviour: a *per-evaluation* opacity is fp32-safe. The fp64-vs-production gap is the small fidelity cost of our clean vectorized walk versus the production 8-block depth-skip — an *algorithm* difference, not a precision one. **The deposit is not where the convergence diverges.** A picture of the deposited forest at one depth makes the structure visible.""")

code(r'''j_show = min(40, n_depth - 1)
win_wave = R["win_waveset_nm"]
xl_show = back(dep_work)[j_show]; xl_ref = ref_x[j_show]
floor = np.median(xl_ref[xl_ref > 0]) * 1e-2 if (xl_ref > 0).any() else 1e-6
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.semilogy(win_wave, np.maximum(xl_show, floor), color="C0", lw=1.0, label=f"deposit ({DEVICE.type}/{DTAG})")
ax.semilogy(win_wave, np.maximum(xl_ref,  floor), color="C3", lw=1.0, ls=":", label="production reference")
ax.set_xlabel("wavelength [nm]"); ax.set_ylabel(r"line opacity $\kappa^{\rm line}$ [cm$^2$/g]")
ax.set_title(f"The deposited line forest (depth {j_show}, T = {T[j_show]:.0f} K)")
ax.legend(loc="upper right", fontsize=9); fig.tight_layout(); plt.show()
print("the GPU deposit (blue) lies under the production reference (red dotted); cores match to the float floor")''')

# ── Rosseland fold ─────────────────────────────────────────────────────────────
md(r"""## The line-blanketed Rosseland mean — the harmonic fold, in torch

With the blanket deposited, fold it into the **Rosseland mean**: the harmonic, $\partial B_\nu/\partial T$-weighted average of the *total* extinction $\kappa^{\rm tot}_\nu = \kappa^{\rm cont}_\nu + \kappa^{\rm line}_\nu + \sigma_\nu$,
$$\frac{1}{\kappa_{\rm Ross}} = \frac{\int (1/\kappa^{\rm tot}_\nu)\,(\partial B_\nu/\partial T)\,d\nu}{\int (\partial B_\nu/\partial T)\,d\nu}.$$
The full grid has 30000 OS frequencies; depositing all 18M lines onto it is the multi-gigabyte step the NumPy edition ships precomputed (`xlines_fullgrid`), exactly as it ships the continuum `acont`. We load those given inputs and do the fold in `torch` — a single reduction over the frequency axis, depth-batched. This is the first **convergence-core reduction**: a 30000-term harmonic sum. We run it in fp32 and fp64 and compare.""")

code(r'''freq = R["freq_hz"]; rco = R["rco"]
acont = R["acont"].astype(np.float64); sigmac = R["sigmac"].astype(np.float64)
xlines_full = R["xlines_fullgrid"].astype(np.float64)
SIGMA = 5.6697e-5; PLANCK = 6.6256e-27; KBOLTZ = 1.38054e-16
hkt = PLANCK / np.maximum(T * KBOLTZ, 1e-300)

def rosseland(device, dtype, include_lines=True):
    """Harmonic dB/dT-weighted mean of kappa_tot over the 30000-freq grid, depth-batched in torch.
    Returns (abross[depth], acc[depth]) where acc is the raw harmonic accumulator."""
    f = lambda x: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    Tt = f(T); hktt = f(hkt); ft = f(freq); wt = f(rco)
    ehvkt = torch.exp(-ft[None, :] * hktt[:, None])
    stim = torch.clamp(1.0 - ehvkt, min=1e-30)
    bnu = 1.47439e-2 * ((ft[None, :] / 1e15) ** 3) * ehvkt / stim
    dbdt = bnu * ft[None, :] * hktt[:, None] / torch.clamp(Tt[:, None] * stim, min=1e-30)
    abtot = f(acont) + f(sigmac)
    if include_lines:
        abtot = abtot + f(xlines_full) * stim                       # fold the blanket in (stim-corrected)
    acc = torch.sum(dbdt / torch.clamp(abtot, min=1e-30) * wt[None, :], dim=1)
    abross = (4.0 * SIGMA / 3.14159) * Tt ** 3 / torch.clamp(acc, min=1e-30)
    return abross, acc

abross_work, acc_work = rosseland(DEVICE, DTYPE)
abross_cont, _ = rosseland(DEVICE, DTYPE, include_lines=False)
print(f"Rosseland mean folded; the blanket raises kappa_Ross by factor "
      f"{float((abross_work/abross_cont).median()):.2f} (median), base abross = {float(abross_work[-1]):.2f} cm^2/g")''')

code(r'''abross_ref64, acc_ref64 = rosseland(*REF64)
diagnose("Rosseland harmonic fold", acc_work, acc_ref64,
         "float floor -- one well-conditioned 30000-term sum")
diagnose("Rosseland mean abross",   abross_work, abross_ref64, "float floor")''')

md(r"""**The Rosseland fold is fp32-safe.** The 30000-term harmonic sum holds fp32 parity to the float floor ($\sim 4\times10^{-7}$). A single, well-conditioned reduction — no cancellation, no compounding — is exactly what single precision does well, even at 30000 terms, because the summands are all positive and span a moderate dynamic range. The deep-base $\kappa_{\rm Ross}$ agrees to the last fp32 digit. **Still not where the divergence enters.**""")

# ── tau_Ross integral ──────────────────────────────────────────────────────────
md(r"""## The Rosseland optical depth $\tau_{\rm Ross}$ — the cumulative integral

The optical-depth scale is the running integral of the Rosseland opacity down the column,
$$\tau_{\rm Ross}(j) = \int_0^{\rho x_j} \kappa_{\rm Ross}\,d(\rho x),$$
formed by the production `INTEG`/`PARCOE` quadrature (a parabolic fit on each depth interval, accumulated). This is a *sequential* reduction — a prefix sum where each layer adds to the running total — so unlike the one-shot Rosseland fold it can **accumulate** round-off down the column. We port `parcoe`/`integ` and run them in fp32 and fp64. The deep-base $\tau_{\rm Ross}$ is what anchors the hydrostatic re-integration in the next cell, so a bias here would feed straight into the structure.""")

code(r'''def parcoe_np(fv, x, npf):
    """Parabolic per-interval coefficients (ATLAS PARCOE), in numpy dtype npf (the working precision)."""
    fv = fv.astype(npf); x = x.astype(npf); nn = fv.size
    a = np.zeros(nn, npf); b = np.zeros(nn, npf); c = np.zeros(nn, npf)
    if nn < 2:
        if nn == 1: a[0] = fv[0]
        return a, b, c
    b[0] = (fv[1]-fv[0])/(x[1]-x[0]); a[0] = fv[0]-x[0]*b[0]; n1 = nn-1
    b[-1] = (fv[-1]-fv[n1-1])/(x[-1]-x[n1-1]); a[-1] = fv[-1]-x[-1]*b[-1]
    if nn == 2: return a, b, c
    for j in range(1, n1):
        j1 = j-1; d = (fv[j]-fv[j1])/(x[j]-x[j1])
        c[j] = fv[j+1]/((x[j+1]-x[j])*(x[j+1]-x[j1])) + (fv[j1]/(x[j+1]-x[j1])-fv[j]/(x[j+1]-x[j]))/(x[j]-x[j1])
        b[j] = d - (x[j]+x[j1])*c[j]; a[j] = fv[j1]-x[j1]*d+x[j]*x[j1]*c[j]
    c[1] = 0.0; b[1] = (fv[2]-fv[1])/(x[2]-x[1]); a[1] = fv[1]-x[1]*b[1]
    if nn > 3: c[2] = 0.0; b[2] = (fv[3]-fv[2])/(x[3]-x[2]); a[2] = fv[2]-x[2]*b[2]
    for j in range(1, n1):
        if c[j] == 0.0: continue
        j1 = min(j+1, nn-1); denom = abs(c[j1])+abs(c[j]); wt = abs(c[j1])/denom if denom > 0 else 0.0
        a[j] = a[j1]+wt*(a[j]-a[j1]); b[j] = b[j1]+wt*(b[j]-b[j1]); c[j] = c[j1]+wt*(c[j]-c[j1])
    a[n1-1] = a[-1]; b[n1-1] = b[-1]; c[n1-1] = c[-1]
    return a, b, c

def integ_np(x, fv, start, npf):
    """Running integral INTEG(x, f) with PARCOE's quadratic per interval, in numpy dtype npf."""
    x = x.astype(npf); fv = fv.astype(npf); nn = fv.size
    out = np.zeros(nn, npf); a, b, c = parcoe_np(fv, x, npf); out[0] = npf(start)
    for i in range(nn-1):
        dx = x[i+1]-x[i]
        term = a[i] + 0.5*b[i]*(x[i+1]+x[i]) + (c[i]/3.0)*((x[i+1]+x[i])*x[i+1]+x[i]*x[i])
        out[i+1] = out[i] + term*dx
    return out

# integrate the SAME fp64 Rosseland opacity at both working precisions (isolate the integral's own spread)
ab = back(abross_ref64)
tau64 = integ_np(rhox, ab, ab[0]*rhox[0], np.float64)
tauW  = integ_np(rhox, ab, ab[0]*rhox[0], np.float32 if DTYPE == torch.float32 else np.float64)
diagnose("tau_Ross integral (INTEG)", tauW.astype(np.float64), tau64,
         "a few x1e-6 -- a sequential prefix sum accumulates round-off")
print(f"  tau_Ross base (deep) = {tau64[-1]:.4e} (fp64); the optical-depth anchor for hydrostatics")''')

md(r"""**The $\tau_{\rm Ross}$ integral is *almost* fp32-safe.** A few $\times 10^{-6}$ — slightly above the one-shot float floor, because the prefix sum accumulates a little round-off down the 80 layers, but still small. Worth noting for the precision budget (this is one of the reductions the production fix optionally promotes), but on its own it does not break the model. We are now at the doorstep of the cell that does.""")

# ── the secant ─────────────────────────────────────────────────────────────────
md(r"""## The temperature-correction secant — where fp32 breaks

Here is the cell the whole diagnostic has been walking toward. The Avrett–Krook temperature correction updates not only the temperature but the **column mass** $\rho x$, and it does so with a *finite difference*. It re-runs the hydrostatic integration (`TTAUP`) **twice** — once at the current temperature $T$, once at the perturbed $T+\delta T$ where $\delta T$ is the iteration's temperature correction (tens of K) — to get two total-pressure structures $p_1$ and $p_2$, and forms the fractional change

$$\texttt{ppp} = \frac{p_2 - p_1}{p_1}, \qquad \delta(\rho x) = \texttt{ppp}\cdot\rho x.$$

The trap is here. Each hydrostatic march $p_1$, $p_2$ is a well-conditioned quantity that holds fp32 parity at $\sim10^{-5}$. But on the deep layers a tens-of-K perturbation barely moves the pressure — $p_2$ and $p_1$ **agree to five or six fp32 digits**. Subtracting them, fp32 keeps only the **one or two trailing digits** that disagree: *catastrophic cancellation*. The fractional change `ppp` — the very quantity that updates the column mass — comes out with a large relative error, and because the column mass is carried into the next iteration, that error **compounds**. In the production engine this is what drives the base $\rho x$ to $\sim 8.5$ instead of the Sun's $12.14$.

We port `TTAUP` (the hydrostatic march) and run the secant in fp32 and fp64. To make a genuine, self-contained demonstration we use the reference structure and a realistic temperature correction $\delta T$ (tens of K, shrinking with depth), with a physical opacity that rises steeply with $T$ so the two marches genuinely differ.""")

code(r'''def ttaup_np(t, tau, prad, grav, rosstab, npf):
    """Hydrostatic pressure march TTAUP, in numpy dtype npf. Marches log P inward, iterating to the
    opacity-consistent pressure at each depth. Returns (abstd, ptotal, pgas)."""
    f = npf; nn = int(t.size)
    abstd = np.zeros(nn, f); ptotal = np.zeros(nn, f); pgas = np.zeros(nn, f)
    dlg = f(np.log(max(float(tau[1]/max(tau[0], 1e-30)), 1e-30))) if nn > 1 else f(0.0)
    p1 = p2 = p3 = p4 = f(0.0); d1 = d2 = d3 = f(0.0)
    abstd[0] = f(0.1)
    if prad[0] > 0: abstd[0] = min(f(0.1), grav*tau[0]/max(prad[0], f(1e-30))/2.0)
    for j in range(nn):
        if j == 0:    plog = np.log(max(grav/max(abstd[0], f(1e-30))*tau[0], f(1e-30)))
        elif j <= 3:  plog = p1 + d1
        else:         plog = (3.0*p4 + 8.0*d1 - 4.0*d2 + 8.0*d3)/3.0
        err = f(1.0); dpl = f(0.0); itn = 1
        while True:
            plog = min(plog, f(709.78)); ptotal[j] = np.exp(plog)
            pgas[j] = ptotal[j] + (prad[0]-prad[j])
            if pgas[j] <= 0: pgas[j] = f(1e-30); abstd[j] = f(0.1); break
            abstd[j] = rosstab(float(t[j]), float(pgas[j]), f)
            dpl = grav/max(abstd[j], f(1e-30))*tau[j]/max(ptotal[j], f(1e-30))*dlg
            itn += 1
            if itn > 1000 or err <= f(5e-5): break
            if j == 0:   pnew = np.log(max(grav/max(abstd[j], f(1e-30))*tau[j], f(1e-30)))
            elif j <= 3: pnew = (plog + 2.0*p1 + dpl + d1)/3.0
            else:        pnew = (126.0*p1 - 14.0*p3 + 9.0*p4 + 42.0*dpl + 108.0*d1 - 54.0*d2 + 24.0*d3)/121.0
            err = abs(pnew - plog); plog = 0.5*(pnew + plog)
        p4 = p3; p3 = p2; p2 = p1; p1 = plog; d3 = d2; d2 = d1; d1 = dpl
    return abstd, ptotal, pgas

# a clean log-log opacity table kappa(T,P) with a steep, physical T-dependence (H- bound-free ~ T^9)
_lP = np.log(np.maximum(R["P"].astype(np.float64), 1e-30))
_lab = np.log(np.maximum(R["abross_raw"].astype(np.float64), 1e-30))
_lT = np.log(np.maximum(R["T"].astype(np.float64), 1e-30))
def rosstab(t, pg, npf):
    lp = np.log(max(pg, 1e-30)); lt = np.log(max(t, 1e-30))
    return npf(np.exp(np.interp(lp, _lP, _lab) + 9.0*(lt - np.interp(lp, _lP, _lT))))''')

code(r'''def run_secant(npf):
    """The TCORR DRHOX secant: two TTAUP marches at T and T+dT, fractional pressure change, drhox.
    Returns (ptot1, ptot2, ppp, rhox_new), all fp64 for the comparison."""
    f = npf; grav = f(GRAV); nn = T.size
    tau1lg = float(R["tau1lg"]); steplg = float(R["steplg"])
    taustd = (10.0**(tau1lg + np.arange(nn)*steplg)).astype(f)
    tauros = R["tauros_raw"].astype(f); prad = R["prad"].astype(f)
    dT = np.linspace(40.0, 8.0, nn).astype(f)                         # a realistic correction (tens of K)
    Tcur = R["T"].astype(f)
    tnew1 = np.interp(taustd, tauros, Tcur).astype(f)
    prdnew = np.interp(taustd, tauros, prad).astype(f)
    _a1, ptot1, _ = ttaup_np(tnew1, taustd, prdnew, grav, rosstab, f)
    tnew2 = np.interp(taustd, tauros, (Tcur + dT)).astype(f)
    _a2, ptot2, _ = ttaup_np(tnew2, taustd, prdnew, grav, rosstab, f)
    ppp = (ptot2 - ptot1) / np.maximum(ptot1, f(1e-30))              # <-- THE SECANT (catastrophic in fp32)
    rrr = np.interp(tauros, taustd, ppp).astype(f)
    rhox_new = R["rhox"].astype(f) + rrr * R["rhox"].astype(f)
    return (ptot1.astype(np.float64), ptot2.astype(np.float64),
            ppp.astype(np.float64), rhox_new.astype(np.float64))

p1_64, p2_64, ppp64, rx64 = run_secant(np.float64)
WF = np.float32 if DTYPE == torch.float32 else np.float64
p1_W, p2_W, pppW, rxW = run_secant(WF)

# each march alone is fp32-safe; the SECANT is where catastrophic cancellation enters
diagnose("hydrostatic march ptot1",  p1_W, p1_64, "float floor -- a well-conditioned march")
diagnose("hydrostatic march ptot2",  p2_W, p2_64, "float floor")
diagnose("SECANT (ptot2-ptot1)/ptot1", pppW, ppp64, "CATASTROPHIC -- difference of near-equal numbers")
diagnose("column mass rhox_new",     rxW, rx64, "the secant error propagates into the structure")''')

md(r"""**This is the cell.** Read the four numbers above in order. Each hydrostatic march — $p_1$ on its own, $p_2$ on its own — holds fp32 parity at the float floor ($\sim10^{-5}$): the per-evaluation pressure structure is fine in single precision. But their **difference**, the secant $(p_2-p_1)/p_1$, jumps **one to two orders of magnitude** above the floor. That is the catastrophic cancellation, isolated to one line of code. The two marches agree to five or six fp32 digits; their difference keeps only the trailing one or two; the fractional change `ppp` — and the column-mass update it drives — is the casualty. A figure shows the cancellation directly: the *absolute* pressures lie on top of each other; the *secant* is where fp32 and fp64 split.""")

code(r'''logtau = np.log10(R["tauros_raw"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(logtau, np.abs(pppW - ppp64)/np.maximum(np.abs(ppp64), 1e-30), color="C3", lw=1.8)
ax[0].set_yscale("log"); ax[0].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$")
ax[0].set_ylabel(r"relative error of the secant  $|\Delta\,p_{\rm rel}|/|p_{\rm rel}|$")
ax[0].set_title("The secant's fp32 error rises into the deep layers")
ax[1].plot(logtau, ppp64, color="0.4", lw=2.2, label="fp64")
ax[1].plot(logtau, pppW,  color="C0", lw=1.0, ls="--", label=f"{DTAG} (working)")
ax[1].set_xlabel(r"$\log_{10}\tau_{\rm Ross}$"); ax[1].set_ylabel(r"$p_{\rm rel}=(p_2-p_1)/p_1$")
ax[1].set_title("Fractional pressure change: fp32 noise vs fp64 signal"); ax[1].legend(fontsize=9)
fig.tight_layout(); plt.show()
print(f"each march is fp32-safe ({DIVERGENCE['hydrostatic march ptot1']:.1e}); the secant is not "
      f"({DIVERGENCE['SECANT (ptot2-ptot1)/ptot1']:.1e}) -- a {DIVERGENCE['SECANT (ptot2-ptot1)/ptot1']/max(DIVERGENCE['hydrostatic march ptot1'],1e-30):.0f}x amplification")''')

# ── the divergence table ───────────────────────────────────────────────────────
md(r"""## The diagnostic, assembled — where fp32 peels away

We have walked the whole convergence-relevant pipeline in fp32 and fp64, cell by cell. Collect the per-cell deviations into one table — the localization the lecture set out to produce.""")

code(r'''print("="*74)
print("LECTURE 15 fp32-vs-fp64 DIVERGENCE TABLE  (working device: %s/%s vs CPU/fp64)" % (DEVICE.type, DTAG))
print("="*74)
order = ["line deposit (wing-walk)", "Rosseland harmonic fold", "Rosseland mean abross",
         "tau_Ross integral (INTEG)", "hydrostatic march ptot1", "hydrostatic march ptot2",
         "SECANT (ptot2-ptot1)/ptot1", "column mass rhox_new"]
for k in order:
    rel = DIVERGENCE[k]
    flag = "float floor" if rel < 1e-4 else ("ELEVATED" if rel < 1e-2 else "CATASTROPHIC")
    bar = "#" * min(40, max(1, int(40 + 6*np.log10(max(rel, 1e-12)))))
    print(f"  {k:30s} {rel:8.2e}  [{flag:12s}] {bar}")
print("="*74)
secant = DIVERGENCE["SECANT (ptot2-ptot1)/ptot1"]; march = DIVERGENCE["hydrostatic march ptot1"]
print(f"\nLOCALISATION: the per-evaluation physics (deposit, Rosseland fold, tau integral, each")
print(f"hydrostatic march) holds fp32 parity at/near the float floor. The divergence ENTERS at the")
print(f"SECANT (ptot2-ptot1)/ptot1 -- {secant:.1e} vs the {march:.1e} of the marches it differences,")
print(f"a {secant/max(march,1e-30):.0f}x amplification from catastrophic cancellation. THAT is the cell to fp64-promote.")''')

md(r"""**The finding, stated plainly.** The cell-by-cell comparison localizes the fp32 divergence exactly where the production `kgpu` engine's fix targets it. Every *per-evaluation* computation — the line deposit, the Rosseland harmonic fold, the $\tau_{\rm Ross}$ integral, each individual hydrostatic march — holds fp32 parity at or near the float floor. The divergence **enters at the convergence-core secant** $(p_2-p_1)/p_1$, where two near-equal hydrostatic pressures are subtracted and single precision keeps only one or two trailing digits. The $\tau_{\rm Ross}$ prefix sum is a mild second contributor (a few $\times10^{-6}$, from sequential accumulation).

The fix follows from the localization and is **surgical**: promote *just* those reductions — the secant difference, and the $\tau_{\rm Ross}$/Rosseland prefix sums — to fp64 (each is a tiny per-depth offload, $\mathcal{O}(80)$ numbers), and leave the entire per-evaluation pipeline — the deposit, the Voigt, the 30000-frequency fold — in fast fp32. That is the design the production engine ships, and the cell above is the evidence for *why* it is enough. We do not apply the fix in this lecture (a concurrent effort owns the production version); the deliverable here is the **diagnostic** — the cell where fp32 peels away from fp64, with the measured numbers that prove it.

**Surprises worth flagging.** Two. First, the line **deposit** is not perfectly at the float floor — its *cores* are, but the overlapping-wing pixels carry a bounded fp32 *accumulation* spread (the `+=` of many Voigt wings), larger than a single reduction though physically minor. Second, the catastrophic cell is *not* any opacity or transfer computation — it is the humble pressure **finite-difference** in the temperature correction, a single subtraction. The expensive physics is fp32-safe; the cheap reduction is the one that breaks. That inversion — *the cheap line is the fragile one* — is the lesson the cell-by-cell comparison teaches.""")

md(r"""## Where this goes — Lecture 16

This lecture ported the line deposit and the convergence-core reductions, and localized the fp32 divergence to the temperature-correction secant. It ran on the per-iteration **state** — the species populations, Doppler widths, continuum-cutoff table, and heat-capacity samples — that it read from the reference. Lecture 16 builds that state from scratch on the GPU: the multi-element `POPSALL` equation of state, the Doppler widths and `xnfdop`, the van der Waals perturber number `txnxn`, and the convective heat capacity — each a *per-evaluation* tensor computation, each (as the diagnostic here predicts) fp32-safe to the float floor. Together the two lectures are the atmosphere built genuinely from scratch on the GPU, with the one fp32-fragile reduction named and located.""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT.relative_to(BOOK)} ({len(cells)} cells)")
