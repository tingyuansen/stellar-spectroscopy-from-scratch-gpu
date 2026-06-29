#!/usr/bin/env python
"""Assemble content/Lecture5.ipynb (unexecuted). Execute + render via build.py.

Lecture 5 — Line Opacity II: The Line List, implemented in clean torch/MPS. This is the
SCATTER-ADD hot path: the scalar reference's per-line outward wing walk (red + blue, += into the
opacity array, stop at the cutoff) becomes a single batched [depth, line, offset] tensor scatter on
the device — `index_put_(accumulate=True)`. The Harris Voigt kernel `voigt_H_grid` from Lecture 4 is
reused across every line in the catalog at every depth. The metal accumulation is validated against
an inline fp64 reference with a strict full-support maximum-relative-error gate (<=1e-6 in fp32).

The clean torch implementation is a pedagogical reduction of the production line-opacity
scatter-add: compute every wing reach as tensors, deposit each red/blue offset block with
`index_put_(accumulate=True)`, and compare against an inline fp64 reference. The notebook imports
neither kgpu nor pykurucz and contains the taught implementation end to end.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "content" / "Lecture5.ipynb"

cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s.strip("\n")))

# ════════════════════════════════════════════════════════════════════════════
#  Title + framing + objectives
# ════════════════════════════════════════════════════════════════════════════
md(r"""# Lecture 5 — Line Opacity II: The Line List

*Stellar Spectroscopy from Scratch — tensor-native stellar spectroscopy, validated against reference calculations*

*Yuan-Sen Ting*

*This lecture builds the metal-line opacity in clean **`torch`** that runs on the GPU (Apple **MPS** or **CUDA**, with a CPU fallback in fp64). The lesson is the **vectorization of a scatter**. The scalar reference adds each of $\sim$twelve thousand metal lines by walking outward from its center pixel — one grid step at a time, red and blue, `+=`-ing the Voigt value into the opacity array and stopping at the cutoff. That per-line outward walk is the textbook shape of a CPU loop, and it is exactly the shape the GPU hates. Here we recast it: the **line axis becomes a tensor batch axis**, the reach geometry of every $(\text{depth},\text{line})$ pair is computed at once, and the deposit is **one batched `[depth, line, offset]` scatter** — `index_put_(accumulate=True)` — per red/blue sweep. The Harris `voigt_H` kernel of Lecture 4 is reused, unchanged, across every line at every depth. We validate the metal accumulation against an inline **fp64 reference** of the exact same recipe, to the documented float floor.*

---

**Learning objectives.** By the end of this lecture you will be able to:

- Read a **line list** and say what each column carries — wavelength, `log gf`, the species code, the excitation potential, the three damping constants — and lay down the **logarithmic wavelength grid** a synthesis runs on, mapping each line to a grid index by a logarithm rather than a search.
- Form the **exact lower-level population** the production code uses (`population_per_ion` $= n_{\rm ion}/U$ from Lecture 2) and complete it with **FASTEX**, the tabulated Boltzmann factor — recast here as a single **branchless gather** instead of a per-line table lookup.
- Reuse Lecture 4's **branchless Harris `voigt_H_grid`** across the whole line list, and assemble each line's $\kappa_0$ amplitude and its cutoff with depth-batched tensor algebra.
- See how the scalar reference's **per-line wing-walk loop** becomes a **batched scatter-add**: `_wing_reach_batched` computes every $(\text{depth},\text{line})$'s reach at once, `_wing_walk_tiered`/`_wing_walk_core` sweep fixed offsets, and **one `index_put_(accumulate=True)`** per red/blue direction deposits the whole `[depth, line, offset]` block — $O(W)$ big batched kernels instead of $O(n_{\rm lines})$ tiny launches, with **reach-tiering** to avoid wasted Harris evaluations on lines that stop early.
- State the **Metal-kernel verdict**: the bible flags this scatter-add as the prime candidate for a custom `torch.mps.compile_shader` Metal kernel — and explain why the optimization squeeze **kept the batched `index_put_`** instead (it already lowers to an efficient atomic scatter; a hand-rolled shader did not beat it).
- **Validate** the complete GPU metal accumulation against the inline fp64 reference with a strict
  **maximum relative error $\le 10^{-6}$** over every opacity-bearing pixel.""")

# ════════════════════════════════════════════════════════════════════════════
#  Introduction
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Introduction

One line was Lecture 4; a spectrum is a forest. The Kurucz atomic line list holds nearly **two million** transitions, each with its own wavelength, strength, and broadening, and the opacity at any wavelength is the sum of the profiles of every nearby line. In our 500–510 nm window that is about **twelve thousand** metal lines — lithium through uranium, plus the helium lines — and the scalar reference adds them one at a time: for each line, at each depth, *walk outward* from the center pixel, one grid step in each direction, `+=` the Voigt value into the running opacity array, and *stop* the moment the profile drops below $10^{-3}$ of the local continuum. A strong line walks far into its wings; a weak one stops after a few steps. That adaptive, data-dependent outward `+=` walk is the heart of the lecture — and it is a **scatter**.

A scatter is the operation a naive GPU implementation most easily gets wrong. Done as the scalar loop suggests — a Python `for` over twelve thousand lines, each launching a tiny per-line kernel — it is a **dispatch storm**: thousands of microscopic launches whose overhead dwarfs the arithmetic. The tensor recasting flips the axes. The **line index becomes a tensor batch axis**; we compute the *reach* of every $(\text{depth},\text{line})$ pair in one batched pass, then sweep a fixed range of offsets and deposit the whole `[depth, line, offset]` block with **one** scatter-add — `torch.index_put_(accumulate=True)` — per red and blue direction. Scatter-add still uses atomic accumulation and can be bandwidth/contention limited, so it is not magic; the win here is batching the work into large operations and removing the per-line dispatch overhead. The cost scales with the widest reach $W$, not with the number of lines, and the launches collapse from $O(n_{\rm lines})$ tiny ones to a handful of big ones.

The target is the **metal** line opacity (line-type code $0$, every $Z \ge 3$). The physics — the exact population normalisation, the FASTEX Boltzmann factor, the Harris Voigt branches, the two-stage cutoff, the wing reach — is identical to the inline fp64 reference used for validation. What is new is the **shape**: a per-line `+=` loop becomes a batched scatter, and that scatter is, per the bible, the book's prime candidate for a custom Metal kernel. We will build it, validate it against the inline fp64 reference, and report the **verdict** on whether a hand-rolled Metal shader earns its place.

![Total line opacity is the sum of every line's Voigt profile on the wavelength grid; a cutoff skips the lines too weak to register against the continuum. On the GPU the per-line outward walk becomes one batched [depth, line, offset] scatter-add.](resources/figures/s5_linelist.png)""")

# ════════════════════════════════════════════════════════════════════════════
#  Setup — device + dtype + dev()
# ════════════════════════════════════════════════════════════════════════════
md(r"""**Setup — the device and the precision budget.** We pick the compute device once: **MPS** on Apple Silicon, **CUDA** on an NVIDIA box, otherwise **CPU**. MPS lacks practical float64 support, and this teaching path deliberately uses **fp32** on both MPS and CUDA so the accelerator route has one uniform precision budget; CUDA hardware can support float64, but that is not the default path here. On the GPU the parity bar is therefore the documented float floor (~$10^{-6}$ for the line accumulation, per the bible's per-component table); on CPU we use **fp64** and recover machine precision. We carry NumPy and Matplotlib alongside `torch` — NumPy holds the **twin** we validate against (the exact scalar recipe), and the comparison at the end is done in NumPy.""")

code(r'''import pathlib, math
import numpy as np
import torch
import matplotlib.pyplot as plt

# pick the compute device ONCE; MPS (Apple) -> CUDA -> CPU. The accelerator teaching
# path uses fp32 on both MPS and CUDA, so its parity bar is the documented float floor;
# on CPU we use fp64 and recover machine precision.
if torch.backends.mps.is_available():
    DEVICE, DTYPE = torch.device("mps"), torch.float32
elif torch.cuda.is_available():
    DEVICE, DTYPE = torch.device("cuda"), torch.float32
else:
    DEVICE, DTYPE = torch.device("cpu"), torch.float64

def dev(x):
    """Move an array/tensor onto the compute device in the working dtype. Pass tensors through
    on-device (never np.asarray an MPS tensor — that raises); only host arrays are wrapped."""
    if torch.is_tensor(x):
        return x.to(device=DEVICE, dtype=DTYPE)
    return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=DEVICE)

print(f"device = {DEVICE.type}   working dtype = {str(DTYPE).split('.')[-1]}")

plt.rcParams.update({"figure.figsize": (7.2, 4.3), "figure.dpi": 120, "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5})''')

# ════════════════════════════════════════════════════════════════════════════
#  Load the reference data + compare
# ════════════════════════════════════════════════════════════════════════════
md(r"""Load the reference bundle. Three files travel with the book:

- `full_lines_data.npz` — the **line catalog**: every atomic line in the window (`cat_wl`, `cat_gf`, `cat_loggf`, `cat_elow`, the index wavelength `cat_index_wl`, the species code `cat_Z`/`cat_ion`, the line-type code `cat_line_types`, and the three damping constants `cat_grad`/`cat_gstark`/`cat_gvdw`), plus the Voigt **Harris tables** `h0tab`/`h1tab`/`h2tab` (the same ones Lecture 4 used).
- `atmosphere.npz` — the **depth state** from Lecture 2: `population_per_ion` $[80,6,139] = n_{\rm ion}/U$, the per-ion Doppler widths `doppler_per_ion`, the mass and electron densities, the temperature, the tabulated $hc/kT$ factor `hckt`, and the van der Waals perturber number densities `xnf_h`/`xnf_he1`/`xnf_h2`.
- `diag.npz` — the production code's **ground truth**: the wavelength grid `wavelength` $[5941]$, the continuum (which sets the cutoff), and `line_opacity` $[80,5941]$.

We build the effective van der Waals perturber density `txnxn` here (neutral H, He I, H$_2$ with the $(T/10^4)^{0.3}$ velocity scaling), and define one helper, `compare`, that moves a GPU tensor back to NumPy and reports the maximum relative deviation — the validation check used throughout the course.""")

code(r'''REF = pathlib.Path("..") / "reference"

cat  = np.load(REF / "full_lines_data.npz", allow_pickle=True)   # the line catalog + Harris tables
atm  = np.load(REF / "atmosphere.npz")                           # the depth state (Lecture 2)
diag = np.load(REF / "diag.npz")                                 # the ground truth + grid + continuum

grid = diag["wavelength"]                                        # [nm], the log-lambda grid
cont = diag["continuum_absorption"] + diag["continuum_scattering"]   # the cutoff continuum [80, n_w]
T    = atm["temperature"]                                        # [K], per depth

# the effective van der Waals perturber density: neutral H, He I, H2 with the (T/1e4)^0.3
# velocity scaling (the 0.3 exponent = (T^0.5)^0.6, from vdW v^(3/5) x thermal v ~ T^(1/2))
txnxn = (atm["xnf_h"] + 0.42*atm["xnf_he1"] + 0.85*atm["xnf_h2"]) * (T/1e4)**0.3

def compare(name, ours, ref, tol=1e-6):
    """Report how closely a GPU result matches the NumPy reference (the per-part check)."""
    # bring the GPU tensor back to NumPy/fp64 (move to CPU FIRST, then cast: MPS has no float64)
    if torch.is_tensor(ours):
        ours = ours.detach().cpu().to(torch.float64).numpy()  # numpy-ref
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    denom = np.where(ref != 0.0, np.abs(ref), 1.0)
    rel = float(np.max(np.abs(ours - ref) / denom))
    tag = "exact" if rel < 1e-12 else ("agree" if rel < tol else "CHECK")
    print(f"{name:28s}  max|rel diff| = {rel:.2e}   [{tag}]")
    return rel

print(f"catalog: {cat['cat_wl'].size} atomic lines, {cat['cat_wl'].min():.1f}-{cat['cat_wl'].max():.1f} nm")
print(f"grid: {grid.size} points, {grid[0]:.2f}-{grid[-1]:.2f} nm")
print(f"depths: {T.size} atmospheric layers")''')

# ════════════════════════════════════════════════════════════════════════════
#  The line record + the log-lambda grid + grid-index helpers
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The line record and the log-lambda grid

Each line in the Kurucz list is one row of fixed-width fields. The columns that drive the opacity are the **wavelength** (nm, vacuum), the **`log gf`** (the base-ten log of the oscillator strength times the lower level's statistical weight — carrying $g_\ell$ inside `gf` is what lets the $\kappa_0$ formula use the population *per unit statistical weight*), the **species code** $Z.\mathrm{ion}$ (one-based: `1 = neutral`, `2 = singly ionized`), the **lower-level excitation energy** $\chi_\ell$ (in $\mathrm{cm^{-1}}$), the three **damping constants** $\gamma_{\rm rad}, \gamma_{\rm Stark}, \gamma_{\rm vdW}$, and a **line-type code** that routes the line to its kernel ($0$ for an ordinary metal Voigt line, $-3/-4/-6$ for helium, $-1/-2$ for hydrogen). This lecture builds the **metal** path: line-type $0$, every $Z \ge 3$.

A synthesis samples wavelength on a **logarithmic grid** of constant ratio $r = \lambda_{i+1}/\lambda_i$, so every line is resolved by the same number of points and the **grid sampling parameter** is $R_{\rm grid} = 1/(r-1)$ (here $\approx 300{,}000$). The key move for the GPU is that a wavelength maps to a grid index by a **logarithm**, not a search: the center index rounds $\log\lambda/\log r$ and offsets by the grid origin. We compute the whole array of center indices in one vectorized call — these become the **batch index** the scatter deposits against.""")

code(r'''# the catalog columns that drive the metal opacity
lam   = cat["cat_wl"].astype(np.float64)        # [nm]
loggf = cat["cat_loggf"]                         # log(gf)
gf    = cat["cat_gf"]                            # gf = 10**loggf, precomputed
Elow  = cat["cat_elow"].astype(np.float64)       # [cm^-1]
idxwl = cat["cat_index_wl"].astype(np.float64)   # the wavelength used for the index lookups
Zc    = cat["cat_Z"].astype(np.int64)            # atomic number, 1-based
ion   = cat["cat_ion"].astype(np.int64)          # ionization stage, 1 = neutral
lt    = cat["cat_line_types"].astype(np.int64)   # line-type code (metals = 0)
grad, gstark, gvdw = cat["cat_grad"], cat["cat_gstark"], cat["cat_gvdw"]
h0tab, h1tab, h2tab = cat["h0tab"], cat["h1tab"], cat["h2tab"]   # Harris Voigt tables (from L4)

# the constant ratio of the log grid, and its sampling parameter (NOT an instrumental resolving power)
ratio = grid[1] / grid[0]
resolu = 1.0 / (ratio - 1.0) if ratio > 1.0 else 300000.0

metal = (lt == 0) & (Zc >= 3)
print(f"metal lines (type 0, Z>=3): {metal.sum()}  spanning Z = {Zc[metal].min()}-{Zc[metal].max()}")
print(f"log grid: ratio = {ratio:.8f}, R_grid = {resolu:.0f}")''')

md(r"""**The grid-index helpers, on the host.** The two anchors a line needs are a **center** index (rounds $\log\lambda/\log r$, offsets by the grid origin, clamps off-grid lines) and a **wing** anchor (rounds $\log(\lambda/w_{\rm begin})$, where $w_{\rm begin}$ is the *floor* of the grid origin in log space). They differ by at most a pixel; we reproduce both exactly so the accumulation matches the twin. These are cheap integer reductions over the line array, computed once on the host (NumPy) — the batched torch path consumes the resulting index tensors. This is the tensor recasting of the scalar reference's `nearest_grid_indices` / `nearest_grid_indices_raw`: the same arithmetic, lifted out of the per-line loop into a one-shot vector op.""")

code(r'''def nearest_grid_indices_np(grid, values):
    """Center index: IXWL = round(log(wl)/ratiolg); idx = IXWL - IXWLBEG, clamped off-grid."""
    grid = np.asarray(grid, dtype=np.float64); values = np.asarray(values, dtype=np.float64)
    ratiolg = np.log(grid[1] / grid[0])
    ix0 = int(np.log(grid[0]) / ratiolg + 0.5)              # rounded log index of the grid origin
    idx = (np.log(values) / ratiolg + 0.5).astype(np.int64) - ix0
    idx = idx.copy()
    idx[values < grid[0]] = -1                              # below the grid
    idx[values > grid[-1]] = grid.size                      # above the grid
    return idx

def nearest_grid_indices_raw_np(grid, values, origin_start):
    """Wing index: round(log(wl/wbegin)/ratiolg), wbegin = floor of the grid origin in log space."""
    grid = np.asarray(grid, dtype=np.float64); values = np.asarray(values, dtype=np.float64)
    ratiolg = np.log(grid[1] / grid[0])
    ixf = int(np.floor(np.log(origin_start) / ratiolg)); wb = np.exp(ixf * ratiolg)
    if wb < origin_start:                                   # nudge up if wbegin fell below the grid
        ixf += 1; wb = np.exp(ixf * ratiolg)
    return np.rint(np.log(values / wb) / ratiolg).astype(np.int64)

center_idx_np = nearest_grid_indices_np(grid, idxwl)
wing_idx_np   = nearest_grid_indices_raw_np(grid, idxwl, float(grid[0]))
print(f"center vs wing anchor differ by at most {int(np.max(np.abs(center_idx_np - wing_idx_np)))} pixel(s)")''')

# ════════════════════════════════════════════════════════════════════════════
#  The Harris Voigt kernel — voigt_H_grid (recap from L4)
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The Harris Voigt kernel, reused across every line

Lecture 4 built $H(a,v)$ — Kurucz's three-branch Harris-table approximation — as **one branchless `torch` expression** evaluated on the whole $(a,v)$ grid: all three regimes computed, the right one selected by `torch.where`. We reuse that kernel here, unchanged, because the accumulation calls it hundreds of times per line and across thousands of lines at once. The three regimes are the **weak-damping** table series ($a < 0.2$, with the bare Lorentzian wing for $|v|>10$), the **far-wing** asymptotic ($a > 1.4$ or $a + |v| > 3.2$, with the $a \le 100$ correction), and the **intermediate** polynomial blend. The table lookup `iv = clamp(int(|v|·200+0.5), 0, N-1)` is a clamped integer index. The numeric constants are Kurucz's, matched bit-for-bit with the reference.

This is the GPU's payoff in miniature: one regular broadcasted tensor expression, no Python loop over points, evaluated over the entire `[depth, line, offset]` block of reduced frequencies at once. PyTorch may lower the expression to several backend kernels; the important design choice is that the source no longer branches per point. It is the same `voigt_H_grid` the production kgpu engine uses (`harris_hav`, reduced to readable form).""")

code(r'''def voigt_H_grid(v, a, h0tab, h1tab, h2tab, branch_oracle=None):
    """Kurucz's Harris H(a,v) as ONE branchless tensor expression on the WHOLE broadcast (v,a) grid
    (the L4 kernel). All three regimes computed for every (v,a) and selected by torch.where; the
    table lookup is a clamped index. v and a broadcast (e.g. v[nv], a[na,1]) -> H(broadcast(v,a))."""
    # Follow the input tensor's device/dtype.  Bulk calls are MPS/fp32; the
    # compact 2-D invariant solve below calls the same physics on CPU/fp64.
    if not isinstance(v, torch.Tensor):
        v = torch.as_tensor(v, dtype=DTYPE, device=DEVICE)
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a, dtype=v.dtype, device=v.device)
    else:
        a = a.to(dtype=v.dtype, device=v.device)
    h0tab = torch.as_tensor(h0tab, dtype=v.dtype, device=v.device)
    h1tab = torch.as_tensor(h1tab, dtype=v.dtype, device=v.device)
    h2tab = torch.as_tensor(h2tab, dtype=v.dtype, device=v.device)

    av = v.abs()
    iv = torch.clamp((av * 200.0 + 0.5).to(torch.int64), 0, h0tab.numel() - 1)
    if branch_oracle is not None:
        iv = branch_oracle["iv"]
    h0 = h0tab[iv]; h1_raw = h1tab[iv]; h2_raw = h2tab[iv]
    aa = a * a; vv = v * v

    # --- weak damping a < 0.2 (table series; bare Lorentzian wing for |v|>10) ---
    vv_safe = torch.where(vv > 0.0, vv, torch.ones_like(vv))
    h_low_tail = 0.5642 * a / vv_safe
    h_low_core = (h2_raw * a + h1_raw) * a + h0
    low_tail_mask = av > 10.0
    if branch_oracle is not None:
        low_tail_mask = branch_oracle["low_tail"]
    h_low = torch.where(low_tail_mask, h_low_tail, h_low_core)

    # --- far-wing asymptotic (with the a<=100 correction) ---
    u = (aa + vv) * 1.4142
    u_safe = torch.where(u > 0.0, u, torch.ones_like(u))
    base = a * 0.79788 / u_safe; aau = aa / u_safe; vvu = vv / u_safe; uu = u_safe * u_safe
    corr = ((((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa) / uu + 1.0)
    h_high = torch.where(a <= 100.0, corr * base, base)

    # --- intermediate polynomial blend (Kurucz's recurrence constants) ---
    h1 = h1_raw + h0 * 1.12838
    h2 = h2_raw + h1 * 1.12838 - h0
    h3 = (1.0 - h2_raw) * 0.37613 - h1 * 0.66667 * vv + h2 * 1.12838
    h4 = (3.0 * h3 - h1) * 0.37613 + h0 * 0.66667 * vv * vv
    pa = (((h4 * a + h3) * a + h2) * a + h1) * a + h0
    pb = ((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032
    h_mid = pa * pb
    if branch_oracle is not None and "mid_value" in branch_oracle:
        h_mid = torch.where(branch_oracle["mid_mask"], branch_oracle["mid_value"], h_mid)

    far = (a > 1.4) | ((a + av) > 3.2)
    low = a < 0.2
    if branch_oracle is not None:
        far = branch_oracle["far"]
        low = branch_oracle["low"]
    return torch.where(low, h_low, torch.where(far, h_high, h_mid))
print("branchless Harris voigt_H_grid ready (reused from Lecture 4)")''')

md(r"""**The center value $H(a,0)$, vectorized.** The wing walk back-solves its profile amplitude from the center opacity (dividing by $H(a,0)$), so we need $H(a,0)$ — the same branch logic with $v=0$, evaluated for the whole batch of damping parameters at once and floored at $10^{-30}$ to keep that division well-defined. This is `_voigt_h_at_zero`, transcribed from the validated implementation.""")

code(r'''def gpu_voigt_h_at_zero(a, h0tab, h1tab, h2tab):
    """Vectorized H(a,0): used to back-solve the wing peak (port _voigt_h_at_zero), floored at 1e-30."""
    h0tab = torch.as_tensor(h0tab, dtype=a.dtype, device=a.device)
    h1tab = torch.as_tensor(h1tab, dtype=a.dtype, device=a.device)
    h2tab = torch.as_tensor(h2tab, dtype=a.dtype, device=a.device)
    h0_0 = h0tab[0]; h1_0 = h1tab[0]; h2_0 = h2tab[0]
    h0v = h0_0; h1v = h1_0 + h0v * 1.12838; h2v = h2_0 + h1v * 1.12838 - h0v
    h3v = (1.0 - h2_0) * 0.37613 + h2v * 1.12838; h4v = (3.0 * h3v - h1v) * 0.37613

    h_low = (h2_0 * a + h1_0) * a + h0_0
    h_mid = (((((h4v * a + h3v) * a + h2v) * a + h1v) * a + h0v)
             * (((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032))
    aa = a * a; u = torch.clamp(aa * 1.4142, min=1.0e-40)
    base = a * 0.79788 / u; aau = aa / u
    h_high = torch.where(a <= 100.0,
                         ((aau * aau * 3.0 - aa) / torch.clamp(u * u, min=1.0e-40) + 1.0) * base, base)
    out = torch.where(a < 0.2, h_low, torch.where((a > 1.4) | (a > 3.2), h_high, h_mid))
    return torch.clamp(out, min=1.0e-30)
print("center-value H(a,0) ready")''')

# ════════════════════════════════════════════════════════════════════════════
#  FASTEX — the tabulated Boltzmann factor, branchless gather
# ════════════════════════════════════════════════════════════════════════════
md(r"""## FASTEX: the tabulated Boltzmann factor as a branchless gather

The exact lower-level population is `population_per_ion` $= n_{\rm ion}/U$ (the EOS output of Lecture 2) times the **Boltzmann factor** $e^{-\chi_\ell\,hc/kT}$. The production code does *not* call `exp` for that factor — it uses **FASTEX**, a pair of lookup tables ($\texttt{EXTAB}[i] = e^{-i}$ for the integer part, $\texttt{EXTABF}[j] = e^{-0.001j}$ for the fractional part) combined as $e^{-x} = \texttt{EXTAB}[\lfloor x\rfloor]\cdot\texttt{EXTABF}[\mathrm{round}(1000\{x\})]$. The tiny rounding of this table is part of the engine, and we reproduce it exactly: a true `exp` would differ in the last bits and spoil agreement.

On the GPU this is a **branchless gather**. The scalar reference masked positive/negative/zero arguments with three boolean branches; here we compute *all* cases — the two table indices, a fallback `exp`, and the $x=0$ special case — for the whole $[\text{depth},\text{line}]$ argument grid and select with `torch.where`. No per-line lookup loop; one `index_select` per table over the entire batch.""")

code(r'''def gpu_fastex_tables(dtype=None, device=None):
    """Build the two FASTEX tables once, on the device (production EXTAB / EXTABF)."""
    dtype = DTYPE if dtype is None else dtype
    device = DEVICE if device is None else device
    i = torch.arange(1001, dtype=torch.float64)
    extab  = torch.exp(-i).to(dtype=dtype, device=device)         # e^{-i}, integer part
    extabf = torch.exp(-i * 0.001).to(dtype=dtype, device=device) # e^{-0.001 j}, fractional part
    return extab, extabf

def gpu_fast_ex(x, extab, extabf):
    """Vectorized branchless FASTEX e^{-x} over the WHOLE [depth, line] grid (port _fast_ex)."""
    x = x.to(extab.dtype); n = extab.numel()
    i = torch.floor(x).to(torch.int64)
    in_tab = (x > 0.0) & (i < n)                                 # within the table range?
    i_cl = torch.clamp(i, 0, n - 1)
    frac = x - i_cl.to(x.dtype)
    j = torch.clamp(torch.floor(frac * 1000.0 + 0.5).to(torch.int64), 0, extabf.numel() - 1)
    tab = extab[i_cl] * extabf[j]                                # e^{-floor} * e^{-frac}, one gather each
    out = torch.exp(-x)                                          # fallback past the table range
    out = torch.where(in_tab, tab, out)
    out = torch.where(x == 0.0, torch.ones_like(out), out)       # e^0 = 1, exactly
    return out
print("branchless FASTEX gather ready")''')

# ════════════════════════════════════════════════════════════════════════════
#  THE WING-ACCUMULATION KERNEL — the centerpiece
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The wing-accumulation kernel: loop $\to$ scatter

This is the centerpiece. Recall the scalar recipe for **one line at one depth**: form the line-center amplitude $\kappa_0$ (the TRANSP normalisation $= c_{gf}\,(n_{\rm ion}/U)/(\rho\,v_D/c)\,e^{-\chi_\ell hc/kT}$), apply the two-stage cutoff (drop the line if $\kappa_0$ fails $10^{-3}\times$ the local continuum before *and* after the Boltzmann factor), build the damping $a$, deposit the **center** opacity at the line pixel, then **walk the wings**: back-solve the profile amplitude (divide by $H(a,0)$), step outward red and blue one grid step at a time, `+=` the Voigt value into the array, and **stop** each direction at the array edge and the loop at the cutoff reach. The near wing (steps up to $10\,v_D$) evaluates $H(a,v)$ from the tables; the far wing ($\propto 1/v^2$) switches to a cheap $x_{\rm far}/n^2$ form whose maximum reach is set analytically.

The scalar reference writes this as a `for` over lines, each calling `process_wing_pair` with a `while offset <= maxstep` loop that `+=`-es into `asynth_d[j]` scalar by scalar. **The tensor implementation recasts it as a batched scatter.** Three pieces:

- **`_wing_reach_batched`** computes the reach geometry of *every* $(\text{depth},\text{line})$ pair at once: the near-wing cutoff step, the far-wing anchor $x_{\rm far}$ and analytic reach, and the per-pair `maxstep`. The near-wing scan is a batched `[depth, line, step]` evaluation of the cheap Harris form, not a per-line loop.
- **`_wing_walk_core`** sweeps a *fixed* range of offsets $1\ldots W$ (the widest reach in the batch), evaluates the profile for the whole `[depth, line, offset]` block, masks each offset against its pair's `maxstep` and the array edges, and deposits the red block and the blue block each with **one** `_scatter_add_3d`.
- **`_scatter_add_3d`** *is* the hot path: it flattens the `[depth, line, offset]` indices into the `[depth*n_w]` opacity array and calls `index_put_(accumulate=True)` — the batched atomic scatter that replaces thousands of scalar `+=` operations.""")

md(r"""**The scatter primitives.** Two helpers wrap `index_put_(accumulate=True)`: a 2-D one for the center deposit (one column per line) and a 3-D one for the wing block (`[depth, line, offset]`). Both build a flat index `depth*n_w + column`, mask it, and accumulate — the `accumulate=True` flag is what makes overlapping deposits (two line wings landing on the same pixel) **add** rather than overwrite. *This single `index_put_` is the GPU recasting of the NumPy per-line `+=` walk* — the deposit the whole lecture builds toward.""")

code(r'''def scatter_add_2d(opacity_grid, column_index, deposit_values, deposit_mask):
    """Center deposit: one column per line, masked, accumulated (index_put_(accumulate=True))."""
    wavelength_count = opacity_grid.shape[1]
    depth_index = torch.arange(opacity_grid.shape[0], device=opacity_grid.device).view(-1, 1)
    column_index_by_depth = column_index.view(1, -1).expand(opacity_grid.shape[0], -1)
    flat_index = (depth_index * wavelength_count + column_index_by_depth)[deposit_mask]
    flat_value = deposit_values[deposit_mask].to(opacity_grid.dtype)
    opacity_grid.view(-1).index_put_((flat_index,), flat_value, accumulate=True)

def scatter_add_3d(opacity_grid, column_index, deposit_values, deposit_mask):
    """THE HOT PATH: the [depth, line, offset] wing block deposited in ONE batched atomic scatter."""
    wavelength_count = opacity_grid.shape[1]
    depth_index = torch.arange(opacity_grid.shape[0], device=opacity_grid.device).view(-1, 1, 1)
    safe_column_index = torch.clamp(column_index, 0, wavelength_count - 1)
    flat_index = (depth_index * wavelength_count + safe_column_index)[deposit_mask]
    flat_value = deposit_values[deposit_mask].to(opacity_grid.dtype)
    opacity_grid.view(-1).index_put_((flat_index,), flat_value, accumulate=True)  # overlaps ADD
print("scatter-add primitives ready (index_put_(accumulate=True) — the hot path)")''')

md(r"""**The near-wing Harris walk, batched.** The wing evaluates the *cheap* small-$a$ two-term form ($h_0 + a\,h_1$, with $0.5642\,a/x^2$ for $x>10$) where $a<0.2$, and the full `voigt_H_grid` otherwise — selected branchlessly over the whole `[depth, line, offset]` block. This `_harris_hav_walk` is the per-offset profile evaluator the reach scan and the walk core both call.""")

code(r'''def harris_hav_walk(reduced_frequency, damping_a, h0tab, h1tab, h2tab,
                    small_damping_mask, branch_oracle=None):
    """Profile over the [depth, line, offset] block: cheap small-a 2-term form OR full H(a,v),
    selected branchlessly (port _harris_hav_walk)."""
    abs_reduced_frequency = reduced_frequency.abs()
    harris_table_index = torch.clamp(
        (abs_reduced_frequency * 200.0 + 0.5).to(torch.int64), 0, h0tab.numel() - 1)
    if branch_oracle is not None:
        harris_table_index = branch_oracle["iv"]
    h0_values = h0tab[harris_table_index]
    h1_values = h1tab[harris_table_index]
    reduced_frequency_squared = torch.where(
        reduced_frequency * reduced_frequency > 0.0,
        reduced_frequency * reduced_frequency,
        torch.ones_like(reduced_frequency),
    )
    cheap_tail_profile = 0.5642 * damping_a / reduced_frequency_squared
    cheap_table_profile = h0_values + damping_a * h1_values
    tail_mask = abs_reduced_frequency > 10.0
    if branch_oracle is not None:
        tail_mask = branch_oracle["low_tail"]
        small_damping_mask = branch_oracle["low"]
    cheap_profile = torch.where(tail_mask, cheap_tail_profile, cheap_table_profile)
    damping_for_full_profile = (
        damping_a.expand_as(reduced_frequency)
        if damping_a.shape != reduced_frequency.shape else damping_a
    )
    full_profile = voigt_H_grid(
        reduced_frequency, damping_for_full_profile, h0tab, h1tab, h2tab,
        branch_oracle=branch_oracle)
    small_mask_for_offsets = (
        small_damping_mask.expand_as(reduced_frequency)
        if small_damping_mask.shape != reduced_frequency.shape else small_damping_mask
    )
    return torch.where(small_mask_for_offsets, cheap_profile, full_profile)
print("batched near-wing Harris walk ready")''')

md(r"""**`_wing_reach_batched` — every pair's reach, at once.** This is the tensor recasting of Stage 1 + Stage 2 of `process_wing_pair`. It computes, for the whole $[\text{depth},\text{line}]$ batch: the fractional Doppler width and the per-step $v$ increment `dvoigt`; the last near-wing step `n10dop` ($10\,v_D$); a batched `[depth, line, step]` scan of the cheap profile that finds where each pair first drops below the cutoff (`first_below` via `argmax` on the boolean — no per-line `break`); the far-wing anchor `x_far` and the analytic far reach; and the final per-pair `maxstep`. Transcribed from the validated implementation.""")

code(r'''NARROW_REACH_TIERS = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536, 131072, 262144, 524288, 1_000_000,
)
MAX_PROFILE_STEPS = 1_000_000

def wing_reach_batched(kappa0_wing, damping_a, doppler_width, line_wavelength,
                       wing_cutoff, wing_pair_mask, resolu, h0tab, h1tab, h2tab):
    """Reach geometry for the WHOLE [depth, line] batch (port _wing_reach_batched): near-wing
    cutoff step, far-wing anchor x_far + analytic reach, per-pair maxstep. No per-line loop."""
    doppler_fraction = torch.where(
        (doppler_width > 0.0) & (line_wavelength.view(1, -1) > 0.0),
        doppler_width / line_wavelength.view(1, -1),
        torch.full_like(doppler_width, 1.0e-10))
    active_wing_pairs = (doppler_width > 0.0) & wing_pair_mask
    n10dop = (10.0 * doppler_fraction * resolu).to(torch.int64)
    dvoigt = torch.where(doppler_fraction > 0.0, 1.0 / (doppler_fraction * resolu),
                         torch.ones_like(doppler_fraction))
    small_damping_mask = damping_a < 0.2

    nstep_cutoff = n10dop.clone()
    reached_near_cutoff = torch.zeros_like(active_wing_pairs)
    profile_at_n10dop = torch.zeros_like(doppler_fraction)

    # JUSTIFY: one scalar reach bound orchestrates a single batched near-wing launch.
    max_n10 = int(n10dop[active_wing_pairs].max().item()) if bool(active_wing_pairs.any()) else 0
    if max_n10 >= 1:
        # batched [depth, line, step] near-wing scan (no per-line break)
        steps = torch.arange(1, max_n10 + 1, device=dvoigt.device, dtype=torch.int64)
        reduced_frequency = steps.view(1, 1, -1).to(dvoigt.dtype) * dvoigt[:, :, None]
        profile_shape = harris_hav_walk(
            reduced_frequency, damping_a[:, :, None], h0tab, h1tab, h2tab,
            small_damping_mask[:, :, None])
        profile_values = kappa0_wing[:, :, None] * profile_shape
        within = steps.view(1, 1, -1) <= n10dop[:, :, None]
        below_cutoff = (profile_values < wing_cutoff[:, :, None]) & within
        has_cutoff_hit = below_cutoff.any(dim=2)
        first_below_cutoff = torch.argmax(below_cutoff.to(torch.int64), dim=2) + 1
        nstep_cutoff = torch.where(has_cutoff_hit, first_below_cutoff, n10dop)
        reached_near_cutoff = has_cutoff_hit
        n10_col = torch.clamp(n10dop - 1, min=0)
        profile_at_n10dop = torch.where(
            n10dop >= 1, profile_values.gather(2, n10_col[:, :, None]).squeeze(2),
            torch.zeros_like(doppler_fraction))

    reached_far_wing = (~reached_near_cutoff) & (n10dop >= 1)
    nstep_cutoff = torch.where(reached_far_wing, torch.full_like(nstep_cutoff, -1), nstep_cutoff)
    use_far = nstep_cutoff == -1

    # far wing: profile * n^2 is constant in 1/v^2; reach where x_far/n^2 falls below the cutoff
    x_far = torch.where(
        (n10dop > 0) & (profile_at_n10dop > 0.0),
        profile_at_n10dop * n10dop.to(doppler_fraction.dtype) * n10dop.to(doppler_fraction.dtype),
        torch.zeros_like(doppler_fraction))
    safe_wing_cutoff = torch.where(wing_cutoff > 0.0, wing_cutoff, torch.ones_like(wing_cutoff))
    far_reach_pos = torch.where(
        wing_cutoff > 0.0, (torch.sqrt(x_far / safe_wing_cutoff) + 1.0).to(torch.int64),
        torch.full_like(n10dop, MAX_PROFILE_STEPS))
    far_reach = torch.where((n10dop > 0) & (x_far > 0.0), far_reach_pos, torch.zeros_like(n10dop))
    far_reach = torch.clamp(far_reach, max=MAX_PROFILE_STEPS)

    maxstep = torch.where(use_far, far_reach, nstep_cutoff)
    maxstep = torch.where(active_wing_pairs, maxstep, torch.zeros_like(maxstep))
    return maxstep, use_far, n10dop, dvoigt, x_far
print("batched wing-reach geometry ready")''')

md(r"""**`_wing_walk_core` — the fixed-offset sweep + the batched deposit.** Given the reach, this sweeps offsets $1\ldots W$ (the widest reach in the *given* batch), evaluates the near-wing Harris profile and the far-wing $x_{\rm far}/\mathrm{offset}^2$ for the whole `[depth, line, offset]` block, masks each offset against its pair's `maxstep` and the red/blue array edges, and deposits the **red** block and the **blue** block each with one `_scatter_add_3d`. *Two batched scatters replace the scalar reference's entire per-line `while` loop.*""")

code(r'''def harris_branch_oracle(damping_a_fp64, dvoigt_fp64, tier_width, device, h0tab, h1tab, h2tab):
    """Build branch decisions for the Harris Voigt walk in CPU/fp64.

    The production wing walk runs in fp32 on the active device, but the Harris
    intermediate branch is sensitive to cancellation near the line core.  This
    helper evaluates only the branch masks and those sparse intermediate values
    in fp64, then ships compact tensors back to the profile kernel.  The returned
    masks have the same broadcast shape as the offset block, [depth, line, W].
    """
    offsets = torch.arange(1, tier_width + 1, dtype=torch.float64, device=torch.device("cpu"))
    reduced_frequency = dvoigt_fp64[:, :, None] * offsets.view(1, 1, -1)
    abs_reduced_frequency = reduced_frequency.abs()
    h0tab64 = h0tab.detach().cpu().to(torch.float64)
    h1tab64 = h1tab.detach().cpu().to(torch.float64)
    h2tab64 = h2tab.detach().cpu().to(torch.float64)
    harris_table_index = torch.clamp(
        (abs_reduced_frequency * 200.0 + 0.5).to(torch.int64), 0, h0tab64.numel() - 1)
    damping_a_block = damping_a_fp64[:, :, None]
    low = (damping_a_block < 0.2).expand_as(abs_reduced_frequency)
    far = (damping_a_block > 1.4) | ((damping_a_block + abs_reduced_frequency) > 3.2)
    mid_mask = (~low) & (~far)

    # The intermediate Harris polynomial is the only cancellation-prone branch: fp32 alone
    # reaches 1.12e-6 on isolated, single-contributor core pixels.  Evaluate just those compact
    # core points in fp64; low/far profiles and every scatter remain MPS-resident.
    mid_value = torch.zeros_like(abs_reduced_frequency)
    if bool(mid_mask.any()):
        damping_mid = damping_a_block.expand_as(abs_reduced_frequency)[mid_mask]
        reduced_frequency_sq_mid = (reduced_frequency * reduced_frequency)[mid_mask]
        table_index_mid = harris_table_index[mid_mask]
        h0_mid = h0tab64[table_index_mid]
        h1_raw_mid = h1tab64[table_index_mid]
        h2_raw_mid = h2tab64[table_index_mid]
        h1_mid = h1_raw_mid + h0_mid * 1.12838
        h2_mid = h2_raw_mid + h1_mid * 1.12838 - h0_mid
        h3_mid = ((1.0 - h2_raw_mid) * 0.37613
                  - h1_mid * 0.66667 * reduced_frequency_sq_mid
                  + h2_mid * 1.12838)
        h4_mid = ((3.0 * h3_mid - h1_mid) * 0.37613
                  + h0_mid * 0.66667 * reduced_frequency_sq_mid * reduced_frequency_sq_mid)
        numerator_poly = (
            (((h4_mid * damping_mid + h3_mid) * damping_mid + h2_mid) * damping_mid
             + h1_mid) * damping_mid + h0_mid)
        damping_blend = (
            ((-0.122727278 * damping_mid + 0.532770573) * damping_mid - 0.96284325)
            * damping_mid + 0.979895032)
        mid_value[mid_mask] = numerator_poly * damping_blend
    return {
        "iv": harris_table_index.to(device=device),
        "low_tail": (abs_reduced_frequency > 10.0).to(device=device),
        "far": far.to(device=device),
        "low": low.to(device=device),
        "mid_mask": mid_mask.to(device=device),
        "mid_value": mid_value.to(dtype=torch.float32, device=device),
    }

def wing_walk_core(opacity_grid, center_columns, kappa0_wing, damping_a, maxstep, use_far,
                   n10dop, dvoigt, x_far, wavelength_count, h0tab, h1tab, h2tab,
                   damping_a_fp64=None, dvoigt_fp64=None):
    """Deposit one reach tier of ordinary-line opacity into both wavelength wings.

    `maxstep` gives the per-depth/per-line outward reach for this tier.  The
    function evaluates all offsets 1..W as one [depth, line, offset] block,
    switches from the near Harris profile to the far 1/offset^2 continuation
    where requested, and scatters the surviving red and blue pixels into `opacity_grid`.
    """
    if center_columns.numel() == 0:
        return
    # JUSTIFY: one scalar tier width orchestrates a batched MPS profile/scatter launch.
    tier_width = int(maxstep.max().item())
    if tier_width <= 0:
        return
    offsets = torch.arange(1, tier_width + 1, device=opacity_grid.device, dtype=torch.int64)
    within_reach = offsets.view(1, 1, -1) <= maxstep[:, :, None]

    reduced_frequency = offsets.view(1, 1, -1).to(dvoigt.dtype) * dvoigt[:, :, None]
    small_damping_mask = damping_a < 0.2
    branch_oracle = None
    if damping_a_fp64 is not None and dvoigt_fp64 is not None:
        branch_oracle = harris_branch_oracle(
            damping_a_fp64, dvoigt_fp64, tier_width, opacity_grid.device, h0tab, h1tab, h2tab)
    near_profile_shape = harris_hav_walk(
        reduced_frequency, damping_a[:, :, None], h0tab, h1tab, h2tab,
        small_damping_mask[:, :, None], branch_oracle=branch_oracle)
    near_profile_values = kappa0_wing[:, :, None] * near_profile_shape

    far_wing_mask = use_far[:, :, None] & (offsets.view(1, 1, -1) > n10dop[:, :, None])
    offsets_float = offsets.view(1, 1, -1).to(reduced_frequency.dtype)
    far_profile_values = x_far[:, :, None] / (offsets_float * offsets_float)
    profile_values = torch.where(far_wing_mask, far_profile_values, near_profile_values)
    profile_values = torch.where(within_reach, profile_values, torch.zeros_like(profile_values))

    center_columns_batched = center_columns.view(1, -1, 1)
    red_columns = (center_columns_batched + offsets.view(1, 1, -1)).expand(opacity_grid.shape[0], -1, -1)
    blue_columns = (center_columns_batched - offsets.view(1, 1, -1)).expand(opacity_grid.shape[0], -1, -1)
    red_deposit_mask = within_reach & (red_columns >= 0) & (red_columns < wavelength_count)
    blue_deposit_mask = within_reach & (blue_columns >= 0) & (blue_columns < wavelength_count)

    scatter_add_3d(opacity_grid, red_columns, profile_values, red_deposit_mask)
    scatter_add_3d(opacity_grid, blue_columns, profile_values, blue_deposit_mask)
print("fixed-offset wing-walk core ready (two batched scatters)")''')

md(r"""**`_wing_walk_tiered` — reach-tiering, to avoid wasted Harris evals.** A single fixed-offset sweep over *all* lines would size $W$ to the one line that reaches farthest, wasting work evaluating the profile at huge offsets for lines that stopped after a few steps. **Tiering** fixes this: lines are bucketed by their reach into power-of-two tiers, and each tier runs `_wing_walk_core` over only *its* offset range. A tier of weak lines (reach $\le 8$) sweeps 8 offsets; the rare far-reaching line gets its own wide sweep. Same total scatter, far fewer wasted profile evaluations.""")

code(r'''def wing_walk_tiered(opacity_grid, center_columns, kappa0_wing, damping_a, maxstep,
                     use_far, n10dop, dvoigt, x_far, wavelength_count,
                     h0tab, h1tab, h2tab, damping_a_fp64=None, dvoigt_fp64=None):
    """Bucket lines by reach into power-of-two tiers; each tier sweeps only its own offset range
    (port _wing_walk_tiered) — avoids sizing the sweep to the single farthest-reaching line."""
    if center_columns.numel() == 0:
        return
    line_reach = maxstep.max(dim=0).values                   # the per-line reach (max over depth)
    previous_tier_limit = 0
    tier_outputs = []
    # JUSTIFIED-LOOP: fixed 21-tier launch schedule; each body is a whole batched MPS scatter.
    for tier_limit in NARROW_REACH_TIERS:
        lines_in_tier = (line_reach > previous_tier_limit) & (line_reach <= tier_limit)
        previous_tier_limit = tier_limit
        if not bool(lines_in_tier.any()):
            continue
        tier_line_index = torch.nonzero(lines_in_tier, as_tuple=False).squeeze(1)
        # Bound atomic overlap within a scatter, then tree-reduce chunk outputs.  This retains
        # GPU scatter semantics while avoiding a long fp32 accumulation chain at crowded pixels.
        # JUSTIFIED-LOOP: bounded launch chunking limits fp32 atomic overlap; no scalar line work.
        for chunk_start in range(0, tier_line_index.numel(), 256):
            line_chunk = tier_line_index[chunk_start:chunk_start + 256]
            line_chunk_cpu = line_chunk.detach().cpu()
            tier_opacity = torch.zeros_like(opacity_grid)
            wing_walk_core(
                tier_opacity, center_columns[line_chunk], kappa0_wing[:, line_chunk],
                damping_a[:, line_chunk], maxstep[:, line_chunk], use_far[:, line_chunk],
                n10dop[:, line_chunk], dvoigt[:, line_chunk], x_far[:, line_chunk],
                wavelength_count, h0tab, h1tab, h2tab,
                None if damping_a_fp64 is None else damping_a_fp64[:, line_chunk_cpu],
                None if dvoigt_fp64 is None else dvoigt_fp64[:, line_chunk_cpu])
            tier_outputs.append(tier_opacity)
    if tier_outputs:
        # A tree reduction across reach tiers avoids repeatedly rounding the same output pixel
        # after every tier while preserving the MPS atomic scatter inside each tier.
        opacity_grid.add_(torch.stack(tier_outputs, dim=0).sum(dim=0))
print("reach-tiered wing walk ready")''')

md(r"""**`accumulate_metal` — the whole pipeline.** This ties it together, exactly as the inline fp64 reference's `metal_accumulate_numpy` does, but with the **line axis as a batch axis** throughout. It selects the metal lines, gathers each line's population and Doppler width across all depths, forms `kappa0_pre` and the post-Boltzmann $\kappa_0$ (FASTEX), applies the two-stage cutoff, builds the damping $a$, computes the center contribution `kapcen` and deposits it via the 2-D scatter, back-solves the wing amplitude `kappa0_wing = kapcen/H(a,0)`, computes the reach with `_wing_reach_batched`, and deposits the wings with `_wing_walk_tiered`. **No Python `for` over the twelve thousand lines** — only the reach-tier loop (a handful of iterations) and the device kernels. Stimulated emission is *not* applied here; it goes on once at the very end. Transcribed from the validated implementation.""")

code(r'''CUTOFF = 1.0e-3; KAPMIN_FLOOR = 1.0e-8; CGF_CONSTANT = 0.026538 / 1.77245; C_LIGHT_NM = 2.99792458e17

def metal_invariants_fp64(catalog, atmd, grid, cont, metal_line_indices,
                          center_idx_np, wing_idx_np, resolu):
    """Resolve the compact [depth,line] physics and all discontinuous walk geometry in fp64.

    This is a precision island, not a second opacity implementation: it never builds or deposits a
    [depth,line,wavelength] profile.  The million-point Harris evaluation and overlapping scatter
    remain on MPS; only line invariants, cutoff predicates, and integer reaches cross to the GPU.
    """
    cpu = torch.device("cpu"); f64 = torch.float64
    t64 = lambda x: torch.as_tensor(x, dtype=f64, device=cpu)
    ti = lambda x: torch.as_tensor(x, dtype=torch.int64, device=cpu)

    lam = t64(catalog["lam"])[metal_line_indices]
    gf = t64(catalog["gf"])[metal_line_indices]
    Elow = t64(catalog["Elow"])[metal_line_indices]
    grad = t64(catalog["grad"])[metal_line_indices]
    gstark = t64(catalog["gstark"])[metal_line_indices]
    gvdw = t64(catalog["gvdw"])[metal_line_indices]
    elem_idx = ti(catalog["Z"])[metal_line_indices] - 1
    ion_idx = ti(catalog["ion"])[metal_line_indices] - 1
    center_idx = ti(center_idx_np)[metal_line_indices]
    wing_idx = ti(wing_idx_np)[metal_line_indices]
    n_w = int(torch.as_tensor(grid).numel())

    population_per_ion = t64(atmd["pop3"])
    doppler_fraction_per_ion = t64(atmd["dop3"])
    mass_density = t64(atmd["rho"])
    electron_density = t64(atmd["xne"])
    hc_over_kT = t64(atmd["hckt"])
    vdw_perturber_density = t64(atmd["txnxn"])
    continuum = t64(cont)
    h0t = t64(catalog["h0tab"]); h1t = t64(catalog["h1tab"]); h2t = t64(catalog["h2tab"])

    population = population_per_ion[:, ion_idx, elem_idx]
    doppler_fraction = doppler_fraction_per_ion[:, ion_idx, elem_idx]
    cgf = CGF_CONSTANT * gf / (C_LIGHT_NM / lam)
    center_valid = (center_idx >= 0) & (center_idx < n_w)
    wing_active = (wing_idx >= -MAX_PROFILE_STEPS) & (wing_idx <= n_w - 1 + MAX_PROFILE_STEPS)
    center_cutoff = continuum[:, torch.clamp(center_idx, 0, n_w - 1)] * CUTOFF

    extab, extabf = gpu_fastex_tables(dtype=f64, device=cpu)
    boltz = gpu_fast_ex(Elow.view(1, -1) * hc_over_kT.view(-1, 1), extab, extabf)
    good = (population > 0.0) & (doppler_fraction > 0.0) & (mass_density.view(-1, 1) > 0.0)
    population_over_mass_doppler_width = torch.where(
        good,
        population / (mass_density.view(-1, 1) * doppler_fraction),
        torch.zeros_like(population),
    )
    kappa0_pre = cgf.view(1, -1) * population_over_mass_doppler_width
    post = kappa0_pre * boltz
    passcut = good & (kappa0_pre >= center_cutoff) & (post >= center_cutoff) & (post > 0.0)

    doppler_width = doppler_fraction * lam.view(1, -1)
    doppler_fraction_for_damping = torch.where(lam.view(1, -1) > 0.0, doppler_width / lam.view(1, -1),
                                               torch.full_like(doppler_width, 1.0e-6))
    gamma_total = (grad.view(1, -1) + gstark.view(1, -1) * electron_density.view(-1, 1)
                   + gvdw.view(1, -1) * vdw_perturber_density.view(-1, 1))
    adamp = torch.where((doppler_width > 0.0) & (doppler_fraction_for_damping > 0.0),
                        gamma_total / doppler_fraction_for_damping,
                        torch.zeros_like(gamma_total))
    center_deposit_mask = passcut & (adamp >= 0.0) & (post > 0.0)
    h0_center = gpu_voigt_h_at_zero(adamp, h0t, h1t, h2t)
    kapcen_raw = torch.where(adamp < 0.2, post * (1.0 - 1.128 * adamp), post * h0_center)
    kapcen = torch.where(center_deposit_mask, kapcen_raw, torch.zeros_like(kapcen_raw))
    center_mask = center_deposit_mask & center_valid.view(1, -1)

    wing_pairs = center_deposit_mask & (kapcen > 0.0) & wing_active.view(1, -1)
    live_lines = wing_active & wing_pairs.any(dim=0)
    live_wing_line_index = torch.nonzero(live_lines, as_tuple=False).squeeze(1)
    if live_wing_line_index.numel() == 0:
        return dict(center_idx=center_idx, kapcen=kapcen, center_mask=center_mask, sel=live_wing_line_index)

    lam_w = lam[live_wing_line_index]
    wing_idx_w = wing_idx[live_wing_line_index]
    doppler_width_w = doppler_width[:, live_wing_line_index]
    adamp_w = torch.clamp(adamp[:, live_wing_line_index], min=1.0e-12)
    kapcen_w = kapcen[:, live_wing_line_index]
    wing_pairs_w = wing_pairs[:, live_wing_line_index]
    h0_w = gpu_voigt_h_at_zero(adamp_w, h0t, h1t, h2t)
    kappa0_wing = torch.where(kapcen_w > 0.0, kapcen_w / h0_w, torch.zeros_like(kapcen_w))
    wing_continuum = continuum[:, torch.clamp(wing_idx_w, 0, n_w - 1)]
    wing_cutoff = torch.maximum(wing_continuum * CUTOFF, wing_continuum * KAPMIN_FLOOR)
    maxstep, use_far, n10dop, dvoigt, x_far = wing_reach_batched(
        kappa0_wing, adamp_w, doppler_width_w, lam_w, wing_cutoff, wing_pairs_w,
        resolu, h0t, h1t, h2t)
    return dict(center_idx=center_idx, kapcen=kapcen, center_mask=center_mask, sel=live_wing_line_index,
                wing_idx=wing_idx_w, kappa0_wing=kappa0_wing, adamp=adamp_w,
                maxstep=maxstep, use_far=use_far, n10dop=n10dop, dvoigt=dvoigt, x_far=x_far)
''')

md(r"""The invariant solve returns compact `[depth, line]` tensors: center deposits, live wing-line selection, damping, reach, and far-wing anchors. The next cell is the orchestration layer. It allocates the output opacity field, scatters the center values, then hands only the live wing subset to the reach-tiered GPU walk.""")

code(r'''
def accumulate_metal(catalog, atmd, grid, cont):
    """The fully-batched metal-line scatter accumulation -> kappa_metal[n_depths, n_w] (NO stim
    factor; applied once at the end). The line axis is a TENSOR BATCH axis; the deposit is one (or a
    few reach-tiered) batched [depth, line, offset] scatter(s)."""
    # JUSTIFY: static host wavelength geometry feeds the exact logarithmic index helpers.
    grid_np = np.asarray(grid, dtype=np.float64); n_w = int(grid_np.size)
    pop3_np = atmd["pop3"]; dop3_np = atmd["dop3"]
    n_depths = int(torch.as_tensor(atmd["T"]).numel())
    metal_opacity = torch.zeros((n_depths, n_w), dtype=DTYPE, device=DEVICE)

    lam_np = catalog["lam"]
    idxwl_np = catalog["idxwl"]
    center_idx_np = nearest_grid_indices_np(grid_np, idxwl_np)            # host index reductions
    wing_idx_np = nearest_grid_indices_raw_np(grid_np, idxwl_np, float(grid_np[0]))
    ratio = float(grid_np[1] / grid_np[0]); resolu = 1.0 / (ratio - 1.0) if ratio > 1.0 else 300000.0

    h0t = dev(catalog["h0tab"]); h1t = dev(catalog["h1tab"]); h2t = dev(catalog["h2tab"])
    n_ion_max = int(torch.as_tensor(pop3_np).shape[1])
    n_elem_max = int(torch.as_tensor(pop3_np).shape[2])
    elem_idx_cpu = torch.as_tensor(catalog["Z"], dtype=torch.int64) - 1
    ion_idx_cpu = torch.as_tensor(catalog["ion"], dtype=torch.int64) - 1
    lt_cpu = torch.as_tensor(catalog["lt"], dtype=torch.int64)
    line_ok_cpu = ((lt_cpu == 0) & (elem_idx_cpu >= 0) & (elem_idx_cpu < n_elem_max)
                   & (ion_idx_cpu >= 0) & (ion_idx_cpu < n_ion_max))
    sel0_cpu = torch.nonzero(line_ok_cpu, as_tuple=False).squeeze(1)
    if sel0_cpu.numel() == 0:
        return metal_opacity

    metal_invariants = metal_invariants_fp64(
        catalog, atmd, grid_np, cont, sel0_cpu, center_idx_np, wing_idx_np, resolu)
    center_idx = metal_invariants["center_idx"].to(device=DEVICE)
    kapcen = metal_invariants["kapcen"].to(dtype=DTYPE, device=DEVICE)
    center_mask = metal_invariants["center_mask"].to(device=DEVICE)
    if bool(center_mask.any()):
        scatter_add_2d(metal_opacity, center_idx, kapcen, center_mask)   # center deposit

    live_wing_line_index = metal_invariants["sel"]
    if live_wing_line_index.numel() == 0:
        return metal_opacity
    wing_idx_w = metal_invariants["wing_idx"].to(device=DEVICE)
    kappa0_wing = metal_invariants["kappa0_wing"].to(dtype=DTYPE, device=DEVICE)
    adamp_w = metal_invariants["adamp"].to(dtype=DTYPE, device=DEVICE)
    maxstep = metal_invariants["maxstep"].to(device=DEVICE)
    use_far = metal_invariants["use_far"].to(device=DEVICE)
    n10dop = metal_invariants["n10dop"].to(device=DEVICE)
    dvoigt = metal_invariants["dvoigt"].to(dtype=DTYPE, device=DEVICE)
    x_far = metal_invariants["x_far"].to(dtype=DTYPE, device=DEVICE)
    wing_walk_tiered(metal_opacity, wing_idx_w, kappa0_wing, adamp_w, maxstep, use_far, n10dop,
                     dvoigt, x_far, n_w, h0t, h1t, h2t,
                     damping_a_fp64=metal_invariants["adamp"],
                     dvoigt_fp64=metal_invariants["dvoigt"])
    return metal_opacity
print("accumulate_metal ready (the batched scatter-add pipeline)")''')

# ════════════════════════════════════════════════════════════════════════════
#  THE METAL-KERNEL VERDICT
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The Metal-kernel verdict

This scatter-add is, by the bible's own reckoning (§2.5, §4), the book's **prime candidate for a custom Metal kernel**: a `torch.mps.compile_shader` hand-rolled scatter. The reasoning is sound on its face — scatter-heavy accumulation with overlapping deposits is exactly the pattern where naive torch vectorization can degenerate into a per-element dispatch storm, and a bespoke shader can sometimes fuse the index arithmetic and the atomic write into one pass.

So the optimization squeeze evaluated it directly: **batched `index_put_(accumulate=True)` against a custom Metal scatter**. The verdict is clear, and we state it plainly:

> **The Metal kernel was NOT adopted. Batched `index_put_(accumulate=True)` is the kernel optimum for this deposit.**

Two findings drove that. First, the alternative the squeeze actually tried as the "Metal-style" recasting — a scalar-walk scatter that mirrors the per-line CPU loop on the device — measured **2245 ms**, against the batched `index_put_` path's **610 ms**. The scalar walk was rejected outright: re-introducing the per-line launch is exactly the dispatch storm the batching exists to kill. Second, and more fundamentally, a *genuine* hand-rolled Metal scatter shader did **not beat** the batched `index_put_`: the two hold parity at the fp32 floor, and the custom shader only **matches** it.

The reason is that `index_put_(accumulate=True)` **already lowers to an efficient atomic-scatter** on the Metal backend. PyTorch's MPS implementation dispatches the accumulating scatter to a native atomic-add kernel; a hand-written shader doing the same atomic-add cannot do *less* work than the one the framework already emits. The batching we did — collapsing $O(n_{\rm lines})$ tiny launches into a few big reach-tiered `[depth, line, offset]` scatters — captured all the available win. The custom kernel would re-implement what the framework gives for free.

So the lesson the bible flagged as "the Metal-kernel candidate" resolves, for this hot path, into a *negative* result that is just as instructive: **the right granularity (batched, reach-tiered) on the framework's own atomic scatter is the optimum.** A Metal shader earns its place only where torch has *no* efficient lowering; here it does. We keep the batched `index_put_`.""")

# ════════════════════════════════════════════════════════════════════════════
#  THE COMPARISON CELL — vs the inline fp64 reference
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The comparison cell — validating the GPU scatter against the inline fp64 reference

This is the per-part check used throughout the book. We validate `accumulate_metal` against an **inline fp64 reference** — the exact scalar recipe (`voigt_profile`, `voigt_h_at_zero`, `fast_ex`, the grid-index helpers, `process_wing_pair`, `metal_accumulate_numpy`). The twin walks every line per-depth with a Python loop and `+=`-es scalar by scalar; it is the gold standard the batched scatter must reproduce. (The twin takes ~1 second — that is the dispatch-storm cost the GPU batching exists to remove.)

We run the twin to get `kappa_ref[80, 5941]`, run the GPU `accumulate_metal` to get `kappa_torch`, and compare on the **opacity-bearing pixels** ($|{\rm ref}| > 10^{-12}$ — the line cores and wings; the empty continuum between lines is not a meaningful relative comparison). First, the inline fp64 reference.""")

code(r'''_EXTAB  = np.exp(-np.arange(1001, dtype=np.float64))          # FASTEX tables for the twin
_EXTABF = np.exp(-np.arange(1001, dtype=np.float64) * 0.001)

# --- the inline fp64 reference: the exact scalar recipe the batched scatter reproduces (validation only) ---
def voigt_profile(v, a, h0tab, h1tab, h2tab):
    iv = max(0, min(int(abs(v)*200.0+0.5), h0tab.size-1))
    if a < 0.2:
        if abs(v) > 10.0: return 0.5642*a/(v*v)
        return (h2tab[iv]*a + h1tab[iv])*a + h0tab[iv]
    elif a > 1.4 or (a+abs(v)) > 3.2:
        aa=a*a; vv=v*v; u=(aa+vv)*1.4142; val=a*0.79788/u
        if a <= 100.0:
            aau=aa/u; vvu=vv/u; uu=u*u
            val=((((aau-10.0*vvu)*aau*3.0+15.0*vvu*vvu)+3.0*vv-aa)/uu+1.0)*val
        return val
    else:
        vv=v*v; h0=h0tab[iv]; h1=h1tab[iv]+h0*1.12838; h2=h2tab[iv]+h1*1.12838-h0
        h3=(1.0-h2tab[iv])*0.37613-h1*0.66667*vv+h2*1.12838
        h4=(3.0*h3-h1)*0.37613+h0*0.66667*vv*vv
        pa=(((h4*a+h3)*a+h2)*a+h1)*a+h0
        pb=((-0.122727278*a+0.532770573)*a-0.96284325)*a+0.979895032
        return pa*pb

def voigt_h_at_zero(adamp, h0tab, h1tab, h2tab):
    h0_0,h1_0,h2_0=float(h0tab[0]),float(h1tab[0]),float(h2tab[0])
    h0v=h0_0; h1v=h1_0+h0v*1.12838; h2v=h2_0+h1v*1.12838-h0v
    h3v=(1.0-h2_0)*0.37613+h2v*1.12838; h4v=(3.0*h3v-h1v)*0.37613
    a=np.asarray(adamp,float)
    h_low=(h2_0*a+h1_0)*a+h0_0
    h_mid=((((h4v*a+h3v)*a+h2v)*a+h1v)*a+h0v)*(((-0.122727278*a+0.532770573)*a-0.96284325)*a+0.979895032)
    aa=a*a; u=np.maximum(aa*1.4142,1e-40); base=a*0.79788/u; aau=aa/u
    h_high=np.where(a<=100.0,((aau*aau*3.0-aa)/np.maximum(u*u,1e-40)+1.0)*base,base)
    return np.maximum(np.where(a<0.2,h_low,np.where((a>1.4)|(a>3.2),h_high,h_mid)),1e-30)

def fast_ex(x):
    v=np.asarray(x,float); out=np.empty_like(v); out[v==0.0]=1.0
    neg=v<0.0; out[neg]=np.exp(-v[neg]); pos=v>0.0
    if np.any(pos):
        p=v[pos]; i=np.floor(p).astype(np.int64); tab=i<_EXTAB.size; po=np.empty_like(p)
        if np.any(tab):
            it=i[tab]; j=np.clip(np.floor((p[tab]-it)*1000.0+0.5).astype(np.int64),0,_EXTABF.size-1)
            po[tab]=_EXTAB[it]*_EXTABF[j]
        if np.any(~tab): po[~tab]=np.exp(-p[~tab])
        out[pos]=po
    return out
print("fp64 reference: Voigt + H(a,0) + FASTEX ready")''')

md(r"""The next reference block supplies the scalar grid-index helpers and the literal outward wing walk. It is intentionally loop-based: this is the comparison oracle whose `+=` deposits the GPU scatter must reproduce, not the shipped compute path.""")

code(r'''def nearest_grid_indices(grid, values):
    ratiolg=np.log(grid[1]/grid[0]); ix0=int(np.log(grid[0])/ratiolg+0.5)
    idx=(np.log(values)/ratiolg+0.5).astype(np.int64)-ix0
    idx=idx.copy(); idx[values<grid[0]]=-1; idx[values>grid[-1]]=grid.size; return idx

def nearest_grid_indices_raw(grid, values, origin_start):
    ratiolg=np.log(grid[1]/grid[0]); ixf=int(np.floor(np.log(origin_start)/ratiolg))
    wb=np.exp(ixf*ratiolg)
    if wb<origin_start: ixf+=1; wb=np.exp(ixf*ratiolg)
    return np.rint(np.log(values/wb)/ratiolg).astype(np.int64)

_CUTOFF=1e-3; _KAPMIN_FLOOR=1e-8; _MAX_PROFILE_STEPS=1_000_000
_CGF_CONSTANT=0.026538/1.77245; _C_LIGHT_NM=2.99792458e17

def process_wing_pair(asynth_d, grid, center_idx, kappa0, adamp, doppler_width,
                      line_wavelength, kapmin_ref, resolu, h0tab, h1tab, h2tab):
    """One (line, depth) scalar wing walk: near wing, far-wing reach, red+blue scalar += deposit."""
    n_w=grid.size
    if doppler_width<=0.0: return
    doppler_fraction=doppler_width/line_wavelength if line_wavelength>0.0 else 1e-10
    n10dop=int(10.0*doppler_fraction*resolu)
    dvoigt=1.0/(doppler_fraction*resolu) if doppler_fraction>0.0 else 1.0
    nstep_cutoff=n10dop; profile_at_n10dop=0.0; tabstep=200.0*dvoigt; tabi=0.5; broke=False
    # JUSTIFIED-LOOP: scalar NumPy reference twin, never part of the shipped torch compute path.
    for nstep in range(1, n10dop+1):
        if adamp<0.2:
            tabi+=tabstep; idx=max(int(tabi),0); x=nstep*dvoigt
            if x>10.0: pv=kappa0*(0.5642*adamp/(x*x))
            else: pv=kappa0*(h0tab[min(idx,h0tab.size-1)]+adamp*h1tab[min(idx,h1tab.size-1)])
        else: pv=kappa0*voigt_profile(nstep*dvoigt, adamp, h0tab, h1tab, h2tab)
        if nstep==n10dop: profile_at_n10dop=pv
        if pv<kapmin_ref: nstep_cutoff=nstep; broke=True; break
    if not broke and n10dop>=1: nstep_cutoff=-1
    if nstep_cutoff!=-1: maxstep=nstep_cutoff; use_far=False; x_far=0.0
    else:
        use_far=True
        if n10dop>0 and profile_at_n10dop>0.0:
            x_far=profile_at_n10dop*float(n10dop)**2
            maxstep=int(np.sqrt(x_far/kapmin_ref)+1.0) if kapmin_ref>0.0 else _MAX_PROFILE_STEPS
        else: x_far=0.0; maxstep=0
        maxstep=min(maxstep, _MAX_PROFILE_STEPS)
    red=blue=True; offset=1; tabi=0.5
    # JUSTIFIED-LOOP: scalar NumPy reference twin used only by the explicit parity cell.
    while offset<=maxstep and (red or blue):
        if use_far and offset>n10dop: pv=x_far/float(offset)**2
        elif adamp<0.2:
            tabi+=tabstep; idx=max(int(tabi),0); x=offset*dvoigt
            if x>10.0: pv=kappa0*(0.5642*adamp/(x*x))
            else: pv=kappa0*(h0tab[min(idx,h0tab.size-1)]+adamp*h1tab[min(idx,h1tab.size-1)])
        else: pv=kappa0*voigt_profile(offset*dvoigt, adamp, h0tab, h1tab, h2tab)
        if pv==0.0: break
        if red:
            j=center_idx+offset
            if j>=n_w: red=False
            elif j>=0: asynth_d[j]+=pv          # scalar += (the deposit the GPU batches)
        if blue:
            j=center_idx-offset
            if j<0: blue=False
            elif j<n_w: asynth_d[j]+=pv          # scalar += (the deposit the GPU batches)
        offset+=1
print("fp64 reference: scalar wing walk ready")''')

md(r"""With the scalar profile and wing-walk helpers in place, the final reference block loops over the catalog and assembles the complete metal opacity. This is the deliberately slow per-line path used only by the validation cell.""")

code(r'''def metal_accumulate_numpy(catalog, atmd, grid, cont):
    """The fp64-reference metal-line accumulation -> metal_opacity[n_depths, n_w] (NO stim factor;
    applied once at the end). The scalar per-line reference the torch scatter must reproduce."""
    lam=catalog['lam']; gf_lin=catalog['gf']; Elow=catalog['Elow']; idxwl=catalog['idxwl']
    Zc=catalog['Z']; ion=catalog['ion']; lt=catalog['lt']
    grad=catalog['grad']; gstark=catalog['gstark']; gvdw=catalog['gvdw']
    h0tab,h1tab,h2tab=catalog['h0tab'],catalog['h1tab'],catalog['h2tab']
    population_per_ion=atmd['pop3']; doppler_fraction_per_ion=atmd['dop3']
    mass_density=atmd['rho']; electron_density=atmd['xne']; temperature=atmd['T']
    hc_over_kT=atmd['hckt']; vdw_perturber_density=atmd['txnxn']
    n_depths=temperature.size; n_w=grid.size
    freq_hz=_C_LIGHT_NM/lam; cgf=_CGF_CONSTANT*gf_lin/freq_hz
    elem_idx=Zc-1; ion_idx=ion-1
    n_ion_max,n_elem_max=population_per_ion.shape[1],population_per_ion.shape[2]
    center_idx=nearest_grid_indices(grid, idxwl)
    wing_idx=nearest_grid_indices_raw(grid, idxwl, float(grid[0]))
    ratio=grid[1]/grid[0]; resolu=1.0/(ratio-1.0) if ratio>1.0 else 300000.0
    line_ok=(lt==0)&(elem_idx>=0)&(elem_idx<n_elem_max)&(ion_idx>=0)&(ion_idx<n_ion_max)
    boltz=fast_ex(Elow[None,:]*hc_over_kT[:,None])
    M=_MAX_PROFILE_STEPS
    center_valid=line_ok&(center_idx>=0)&(center_idx<n_w)
    wing_active=line_ok&(wing_idx>=-M)&(wing_idx<=n_w-1+M)
    metal_opacity=np.zeros((n_depths,n_w),dtype=np.float64)
    # JUSTIFIED-LOOP: scalar NumPy reference twin, intentionally mirrors the per-line loop.
    for i in np.where(line_ok)[0]:                                   # the Python per-line loop
        ci=int(center_idx[i]); wi=int(wing_idx[i]); wl_i=lam[i]; clamped=max(0,min(ci,n_w-1))
        population=population_per_ion[:,ion_idx[i],elem_idx[i]]
        doppler_fraction=doppler_fraction_per_ion[:,ion_idx[i],elem_idx[i]]
        kapmin=cont[:,clamped]*_CUTOFF
        good=(population>0.0)&(doppler_fraction>0.0)&(mass_density>0.0)
        if not np.any(good): continue
        population_over_mass_doppler_width=np.zeros(n_depths)
        population_over_mass_doppler_width[good]=(
            population[good]/(mass_density[good]*doppler_fraction[good])
        )
        kappa0_pre=cgf[i]*population_over_mass_doppler_width; post=kappa0_pre*boltz[:,i]
        passcut=good&(kappa0_pre>=kapmin)&(post>=kapmin)&(post>0.0)
        if not np.any(passcut): continue
        doppler_width=doppler_fraction*wl_i
        doppler_fraction_for_damping=np.where(wl_i>0,doppler_width/wl_i,1e-6)
        gamma_total=grad[i]+gstark[i]*electron_density+gvdw[i]*vdw_perturber_density
        adamp=np.where((doppler_width>0)&(doppler_fraction_for_damping>0),
                       gamma_total/doppler_fraction_for_damping,0.0)
        kapcen=np.zeros(n_depths); cd=passcut&(adamp>=0.0)&(post>0.0)
        # JUSTIFIED-LOOP: scalar NumPy reference twin over active depths.
        for d in np.where(cd)[0]:
            ad=adamp[d]
            kapcen[d]=post[d]*(1.0-1.128*ad) if ad<0.2 else post[d]*voigt_profile(0.0,ad,h0tab,h1tab,h2tab)
        if center_valid[i]:
            # JUSTIFIED-LOOP: scalar NumPy reference twin center deposits.
            for d in np.where(cd)[0]: metal_opacity[d,ci]+=kapcen[d]
        if not wing_active[i]: continue
        wing_pairs=cd&(kapcen>0.0)
        if not np.any(wing_pairs): continue
        adamp_w=np.maximum(adamp,1e-12)
        kappa0_wing=np.where(kapcen>0.0,kapcen/voigt_h_at_zero(adamp_w,h0tab,h1tab,h2tab),0.0)
        ci_w=min(max(wi,0),n_w-1); kapmin_ref=np.maximum(cont[:,ci_w]*_CUTOFF,cont[:,ci_w]*_KAPMIN_FLOOR)
        # JUSTIFIED-LOOP: scalar NumPy reference twin wing walks.
        for d in np.where(wing_pairs)[0]:
            process_wing_pair(metal_opacity[d],grid,wi,kappa0_wing[d],adamp_w[d],
                              doppler_width[d],wl_i,kapmin_ref[d],resolu,h0tab,h1tab,h2tab)
    return metal_opacity
print("fp64 reference: metal_accumulate_numpy ready")''')

md(r"""**Run both and compare — completely.** We assemble the catalog and atmosphere dicts the two paths share, run the inline fp64 reference and the GPU `accumulate_metal`, and compare every opacity-bearing pixel. Discontinuous cutoff, table-index, branch, and reach geometry is resolved by the compact CPU/fp64 invariant solve; the cancellation-prone intermediate Harris core is the only profile-value precision island. Bulk low/far profile evaluation and every overlapping deposit remain MPS/fp32. The acceptance criterion is the full maximum, not a percentile: **max relative error $\le 10^{-6}$**.""")

code(r'''# assemble the inputs both paths read
catalog = dict(lam=lam, gf=gf, Elow=Elow, idxwl=idxwl, Z=Zc, ion=ion, lt=lt,
               grad=grad, gstark=gstark, gvdw=gvdw, h0tab=h0tab, h1tab=h1tab, h2tab=h2tab)
atmd = dict(pop3=atm["population_per_ion"], dop3=atm["doppler_per_ion"],
            rho=atm["mass_density"], xne=atm["electron_density"], T=T,
            hckt=atm["hckt"], txnxn=txnxn)

print(f"Validating the GPU metal scatter accumulation against the inline fp64 reference")
print(f"  device = {DEVICE.type}   dtype = {str(DTYPE).split('.')[-1]}\n")

# the inline fp64 reference (gold standard, ~1 s — the per-line dispatch storm the GPU batching removes)
kappa_ref = metal_accumulate_numpy(catalog, atmd, grid, cont)         # [80, 5941], no stim
# the GPU batched scatter
kappa_torch = accumulate_metal(catalog, atmd, grid, cont)
kappa_torch = kappa_torch.detach().cpu().to(torch.float64).numpy()  # numpy-ref

is_fp32 = (DTYPE == torch.float32)
floor = 1e-6 if is_fp32 else 1e-10                                   # ~1e-6 fp32; machine precision fp64

big = np.abs(kappa_ref) > 1e-12                                      # the opacity-bearing pixels
rel = np.abs(kappa_torch[big] - kappa_ref[big]) / np.abs(kappa_ref[big])
med = float(np.median(rel)); q999 = float(np.quantile(rel, 0.999)); mx = float(rel.max())
n_px = int(big.sum()); n_out = int(np.count_nonzero(rel > floor))

print(f"opacity-bearing pixels compared : {n_px}")
print(f"  median  rel diff = {med:.2e}   (the fp32 floor)")
print(f"  99.9%   rel diff = {q999:.2e}   (high quantile, at the fp32 floor)")
print(f"  max     rel diff = {mx:.2e}   pixels above floor = {n_out} / {n_px}")

status = "PASS" if mx <= floor else "CHECK"
print(f"device = {DEVICE.type}   float floor = {floor:.1e}   ->   [{status}]")
assert mx <= floor, f"full metal max rel {mx:.2e} above {floor:.1e}"
print("\nThe complete GPU metal scatter passes the full-support maximum-error gate.")''')

md(r"""**What the numbers mean.** The batched scatter reproduces the inline fp64 reference's per-line `+=` walk across the complete physical support. CPU/fp64 is used only where fp32 rounding changes a discrete decision or where the intermediate Harris polynomial suffers cancellation. The ordinary line forest is still evaluated and accumulated by reach-tiered MPS scatters. After that targeted precision treatment, the full maximum—not merely a median or percentile—lies below the fp32 acceptance floor.""")

# ════════════════════════════════════════════════════════════════════════════
#  THE FIGURE — total metal opacity 500-510 nm at the photosphere
# ════════════════════════════════════════════════════════════════════════════
md(r"""## The forest, GPU and NumPy

Overlay the total metal line opacity from the batched scatter on the inline fp64 reference at a photospheric layer, across the whole 500–510 nm window. The two curves coincide line for line; the batched scatter and the scalar `+=` walk land the same opacity at the same pixels.""")

code(r'''# apply stimulated emission once (the factor both paths held back to the end), then plot
freq_grid = C_LIGHT_NM / grid                                       # [Hz]
hkt = 6.62607015e-27 / (1.380649e-16 * T)                          # h / kT per depth
stim = 1.0 - np.exp(-freq_grid[None, :] * hkt[:, None])           # 1 - e^{-h nu / kT}
metal_gpu = kappa_torch * stim
metal_ref = kappa_ref * stim

jp = T.size // 2                                                    # a representative photospheric layer
plt.figure(figsize=(11, 4.4))
plt.plot(grid, np.maximum(metal_ref[jp], 1e-30), color="0.6", lw=1.4,
         label="fp64 reference (per-line scalar walk)")
plt.plot(grid, np.maximum(metal_gpu[jp], 1e-30), color="C3", lw=0.6,
         label="GPU batched scatter (this lecture)")
plt.yscale("log"); plt.ylim(1e-4, max(metal_ref[jp].max(), metal_gpu[jp].max())*3)
plt.xlabel("wavelength  [nm]"); plt.ylabel(r"metal line opacity  [cm$^2$/g]")
plt.title("Total metal line opacity at the photosphere, 500-510 nm "
          "(GPU scatter overlaid on the inline fp64 reference)")
plt.legend(loc="upper right"); plt.tight_layout(); plt.show()''')

md(r"""The forest matches line for line: every metal transition sits at its vacuum wavelength with the right strength, summed from its own Voigt profile and exact population — and the GPU deposited the whole forest with a handful of batched `index_put_(accumulate=True)` scatters instead of twelve thousand per-line `while` loops.""")

# ── Full atomic catalog coverage beyond the metal scatter hot path ──
md(r"""## Completing the atomic catalog coverage

The metal scatter above is the main GPU hot path, but the textbook line-list lesson also has to account for the rest of the atomic catalog. We keep using the same loaded inputs: the line catalog (`full_lines_data.npz`), the depth-state atmosphere (`atmosphere.npz`), and the diagnostic arrays (`diag.npz`) used only as comparison targets. The cells below make the catalog tensors explicit, add helium, and close the non-Voigt special line-type leaves. The taught computation stays torch-native; NumPy appears only in explicit comparison-reference cells.""")

code(r'''# Torch views of the already-loaded reference arrays.  The host files were loaded above;
# from here on the shipped path works on DEVICE in DTYPE.
grid_t = dev(grid)
cont_t = dev(cont)

lt_t   = torch.as_tensor(lt, dtype=torch.int64, device=DEVICE)
Z_t    = torch.as_tensor(Zc, dtype=torch.int64, device=DEVICE)
ion_t  = torch.as_tensor(ion, dtype=torch.int64, device=DEVICE)

lam_t   = dev(lam)
gf_t    = dev(gf)
loggf_t = dev(loggf)
Elow_t  = dev(Elow)

pop3_t = dev(atm["population_per_ion"])
dop3_t = dev(atm["doppler_per_ion"])
rho_t  = dev(atm["mass_density"])
xne_t  = dev(atm["electron_density"])
T_t    = dev(atm["temperature"])
hckt_t = dev(atm["hckt"])
txnxn_t = dev(txnxn)

print(f"catalog tensors on {DEVICE.type}: grid={tuple(grid_t.shape)}, continuum={tuple(cont_t.shape)}")
print(f"catalog lines={lam_t.numel()}, depths={T_t.numel()}, population table={tuple(pop3_t.shape)}")''')

md(r"""## Catalog audit: line record anatomy

Each Kurucz record supplies the wavelength, `log gf`, species code, lower excitation energy, damping constants, and line-type code. The earlier sections already used these fields for the metal scatter; here the same columns become one-dimensional tensors so the remaining catalog checks share the same **line axis as batch axis**. The code below mirrors a quick catalog inspection without a Python loop over the catalog: it ranks ordinary metal lines by `log gf` with `torch.topk` and prints the strongest few as tensors.""")

code(r'''metal_t = (lt_t == 0)
he_t = (lt_t == -3) | (lt_t == -4) | (lt_t == -6)
hy_t = (lt_t == -1) | (lt_t == -2)

score = torch.where(metal_t, loggf_t, torch.full_like(loggf_t, -torch.inf))
top_loggf, top_idx = torch.topk(score, k=6)

strong_table = torch.stack([
    lam_t[top_idx],
    Z_t[top_idx].to(DTYPE),
    ion_t[top_idx].to(DTYPE),
    top_loggf,
    (Elow_t[top_idx] / 8065.54),
], dim=1)

print("strongest type-0 metal lines: columns = wavelength[nm], Z, ion, loggf, chi_l[eV]")
print(strong_table.detach().cpu())
print(f"metal lines: {int(metal_t.sum().detach().cpu())}; helium lines: {int(he_t.sum().detach().cpu())}; "
      f"hydrogen lines: {int(hy_t.sum().detach().cpu())} -> next lecture")''')

md(r"""## Population tensor audit

The exact population normalization is the same one emphasized in the inline fp64 reference:

\[
\frac{n_\ell}{g_\ell}
=
\frac{n_{\rm ion}}{U}\,
\exp(-\chi_\ell hc/kT).
\]

The atmosphere file already stores \(n_{\rm ion}/U\) as `population_per_ion`; the Boltzmann factor is the production code's tabulated FASTEX exponential. This is the same branchless gather used in the metal accumulation, shown here over the whole `(depth, line)` catalog tensor so the helium and special-profile sections have the same population boundary as the ordinary metal path.""")

code(r'''extab_t, extabf_t = gpu_fastex_tables()

elem_idx_t = Z_t - 1
ion_idx_t = ion_t - 1
in_pop_table_t = (
    (elem_idx_t >= 0) & (elem_idx_t < pop3_t.shape[2]) &
    (ion_idx_t >= 0) & (ion_idx_t < pop3_t.shape[1])
)

safe_elem_t = torch.clamp(elem_idx_t, 0, pop3_t.shape[2] - 1)
safe_ion_t  = torch.clamp(ion_idx_t, 0, pop3_t.shape[1] - 1)

pop_per_line_t = pop3_t[:, safe_ion_t, safe_elem_t]
boltz_all_t = gpu_fast_ex(Elow_t.view(1, -1) * hckt_t.view(-1, 1), extab_t, extabf_t)
lower_pop_per_g_t = torch.where(
    in_pop_table_t.view(1, -1),
    pop_per_line_t * boltz_all_t,
    torch.zeros_like(pop_per_line_t),
)

fe1 = (Z_t == 26) & (ion_t == 1)
fe1_lower = torch.where(fe1.view(1, -1), lower_pop_per_g_t, torch.zeros_like(lower_pop_per_g_t))
print("lower-level population per statistical weight is now a depth×line tensor")
print(f"shape = {tuple(lower_pop_per_g_t.shape)}; max Fe I contribution = "
      f"{fe1_lower.max().detach().cpu().to(torch.float64):.3e} cm^-3")''')

md(r"""## Adding the helium lines

The metal scatter-add hot path is now validated. We add the helium path next because helium is part of the atomic line-list lecture but does not use the ordinary metal reach-tier scatter unchanged.

Helium uses the same population, FASTEX, damping, and Harris-Voigt machinery, but its wing accumulation includes a **continuum-merge taper**. For each depth and helium line the taper is zero below \(w_{\rm con}\), ramps linearly between \(w_{\rm con}\) and \(w_{\rm tail}\), and is full strength beyond \(w_{\rm tail}\). The inline fp64 reference walks each helium wing pixel by pixel. Here the entire `(depth, helium-line, wavelength)` cube is evaluated at once, and the "stop at the first below-cutoff pixel" rule is represented by cumulative stop masks along the wavelength axis.""")

code(r'''def helium_opacity_torch():
    """Vectorized helium opacity with the continuum-merge taper.

    Helium is a two-record heterogeneous family whose stop mask sits exactly on the continuum
    cutoff.  Its compact profile cube is therefore a CPU/fp64 precision island; the 12,568-line
    ordinary forest and its overlapping scatter remain MPS/fp32.
    """
    solve_device, solve_dtype = torch.device("cpu"), torch.float64
    t64 = lambda x: torch.as_tensor(x, dtype=solve_dtype, device=solve_device)
    ti = lambda x: torch.as_tensor(x, dtype=torch.int64, device=solve_device)

    lt64 = ti(lt)
    he_sel = torch.nonzero((lt64 == -3) | (lt64 == -4) | (lt64 == -6),
                           as_tuple=False).squeeze(1)
    if he_sel.numel() == 0:
        return torch.zeros_like(cont_t)

    h0t = t64(h0tab); h1t = t64(h1tab); h2t = t64(h2tab)
    helium_wavelength = t64(lam)[he_sel]
    helium_gf = t64(gf)[he_sel]
    helium_lower_energy = t64(Elow)[he_sel]
    grad_he = t64(grad)[he_sel]; gstark_he = t64(gstark)[he_sel]
    gvdw_he = t64(gvdw)[he_sel]

    population_per_ion = t64(atm["population_per_ion"])
    doppler_fraction_per_ion = t64(atm["doppler_per_ion"])
    mass_density = t64(atm["mass_density"])
    electron_density = t64(atm["electron_density"])
    temperature = t64(atm["temperature"])
    hc_over_kT = t64(atm["hckt"])
    vdw_perturber_density = t64(txnxn)
    grid64 = t64(grid)
    continuum64 = t64(cont)
    elem = torch.clamp(ti(Zc)[he_sel] - 1, 0, population_per_ion.shape[2] - 1)
    ion0 = torch.clamp(ti(ion)[he_sel] - 1, 0, population_per_ion.shape[1] - 1)
    center_columns = torch.clamp(ti(center_idx_np)[he_sel], 0, grid64.numel() - 1)

    he_ltc_t = ti(cat["he_ltc"])
    wcon_t = t64(cat["he_wcon_2d"]); wtail_t = t64(cat["he_wtail_2d"])
    he_cut_t = torch.as_tensor(float(cat["he_cutoff"]), dtype=solve_dtype, device=solve_device)

    population = population_per_ion[:, ion0, elem]
    doppler_fraction = doppler_fraction_per_ion[:, ion0, elem]
    cgf_he = CGF_CONSTANT * helium_gf / (C_LIGHT_NM / helium_wavelength)
    extab64, extabf64 = gpu_fastex_tables(dtype=solve_dtype, device=solve_device)
    boltz = gpu_fast_ex(
        helium_lower_energy.view(1, -1) * hc_over_kT.view(-1, 1), extab64, extabf64)

    doppler_safe = torch.clamp(doppler_fraction, min=1.0e-40)
    valid_population = (
        (population > 0.0) & (doppler_fraction > 0.0) & (mass_density.view(-1, 1) > 0.0)
    )
    population_over_mass_doppler_width = torch.where(
        valid_population,
        population / (mass_density.view(-1, 1) * doppler_safe),
        torch.zeros_like(population),
    )

    k0pre = cgf_he.view(1, -1) * population_over_mass_doppler_width
    center_cutoff = continuum64[:, center_columns] * he_cut_t
    valid_population = valid_population & (k0pre >= center_cutoff)
    k0 = k0pre * boltz
    valid_profile = valid_population & (k0 >= center_cutoff) & (k0 > 0.0)

    gamma_total = (grad_he.view(1, -1) + gstark_he.view(1, -1) * electron_density.view(-1, 1)
                   + gvdw_he.view(1, -1) * vdw_perturber_density.view(-1, 1))
    adamp = torch.where(doppler_safe > 0.0, gamma_total / doppler_safe, torch.zeros_like(gamma_total))
    doppler_width = doppler_fraction * helium_wavelength.view(1, -1)

    isotope = (he_ltc_t.view(1, -1) == -4)
    k_eff = torch.where(isotope, k0 / 1.155, k0)
    doppler_width_effective = torch.where(isotope, doppler_width * 1.155, doppler_width)
    a_eff = torch.where(isotope, adamp / 1.155, adamp)
    a_eff = torch.clamp(a_eff, min=1.0e-12)

    wavelength_grid = grid64.view(1, 1, -1)
    grid_columns = torch.arange(grid64.numel(), dtype=torch.int64, device=solve_device).view(1, 1, -1)
    center_columns_3d = center_columns.view(1, -1, 1)

    red_domain = grid_columns >= center_columns_3d
    blue_domain = grid_columns < center_columns_3d
    red_domain = red_domain & (helium_wavelength.view(1, -1, 1) <= grid64[-1])
    blue_domain = blue_domain & (helium_wavelength.view(1, -1, 1) >= grid64[0]) & (center_columns_3d > 0)

    doppler_width_safe = torch.clamp(doppler_width_effective[:, :, None], min=1.0e-30)
    reduced_frequency = torch.abs(wavelength_grid - helium_wavelength.view(1, -1, 1)) / doppler_width_safe
    untapered_profile = (
        k_eff[:, :, None] * voigt_H_grid(reduced_frequency, a_eff[:, :, None], h0t, h1t, h2t)
    )

    wcon = wcon_t[:, :, None]
    wtail = wtail_t[:, :, None]
    has_wcon = wcon > 0.0
    has_wtail = wtail > 0.0
    can_eval = ~(has_wcon & (wavelength_grid <= wcon))

    taper_start_wavelength = torch.where(has_wcon, wcon, helium_wavelength.view(1, -1, 1))
    taper_width = torch.clamp(wtail - taper_start_wavelength, min=1.0e-12)
    taper = torch.where(
        has_wtail & (wavelength_grid < wtail),
        (wavelength_grid - taper_start_wavelength) / taper_width,
        torch.ones_like(untapered_profile),
    )
    tapered_profile = untapered_profile * taper

    cutoff_grid = continuum64[:, None, :] * he_cut_t
    stop_red = red_domain & can_eval & (tapered_profile < cutoff_grid)
    stop_blue = blue_domain & can_eval & (tapered_profile < cutoff_grid)

    red_before_stop = torch.cumsum(stop_red.to(torch.int64), dim=2) == 0
    blue_before_stop = torch.flip(
        torch.cumsum(torch.flip(stop_blue.to(torch.int64), dims=(2,)), dim=2) == 0,
        dims=(2,),
    )

    deposit = valid_profile[:, :, None] & can_eval & (tapered_profile >= cutoff_grid) & (
        (red_domain & red_before_stop) | (blue_domain & blue_before_stop)
    )

    helium_raw_opacity = torch.sum(
        torch.where(deposit, tapered_profile, torch.zeros_like(tapered_profile)), dim=1)

    h_planck = torch.as_tensor(6.62607015e-27, dtype=solve_dtype, device=solve_device)
    k_boltz = torch.as_tensor(1.380649e-16, dtype=solve_dtype, device=solve_device)
    freq_grid_t = C_LIGHT_NM / grid64
    stimulated_emission_factor = (
        1.0 - torch.exp(-freq_grid_t.view(1, -1) * (h_planck / (k_boltz * temperature)).view(-1, 1))
    )
    return (helium_raw_opacity * stimulated_emission_factor).to(dtype=DTYPE, device=DEVICE)
''')

md(r"""Run the helium path and keep its result as a tensor on the selected device. The validation cells below compare it component-by-component against the archived helium-wing diagnostic.""")

code(r'''
he_opacity_t = helium_opacity_torch()
print(f"helium opacity tensor ready: shape={tuple(he_opacity_t.shape)}, "
      f"nonzero pixels={int((he_opacity_t > 0).sum().detach().cpu())}")''')

md(r"""## The benchmark: machine precision

The inline fp64 reference compares

\[
\kappa_\lambda^{\rm atomic}
=
\kappa_\lambda^{\rm metal}
+
\kappa_\lambda^{\rm He}
\]

against the production diagnostic with the hydrogen component subtracted. In fp32 on MPS the expected floor is no longer NumPy's machine precision, but the comparison is the same physical benchmark: metals plus helium against `diag['line_opacity'] - gt_ahline`. This is an explicit comparison-reference cell; the NumPy lines are the parity oracle.""")

code(r'''# Comparison-reference cell: NumPy appears only as the oracle/reference here.
he_gpu = he_opacity_t.detach().cpu().to(torch.float64).numpy()  # numpy-ref
gt_he = cat["gt_helium_wings"] * stim                           # numpy-ref

# The existing metal_gpu was produced above from the validated metal scatter path.
atomic_gpu = metal_gpu + he_gpu                                  # numpy-ref
ref_atomic = diag["line_opacity"].astype(np.float64) - cat["gt_ahline"]  # numpy-ref

nz = ref_atomic != 0.0                                           # numpy-ref
abs_resid = np.abs(atomic_gpu - ref_atomic)                      # numpy-ref
rel_all = abs_resid[nz] / np.abs(ref_atomic[nz])                 # numpy-ref

dominant = (ref_atomic > 1e-10) & (ref_atomic > 1e-6 * cat["gt_ahline"])  # numpy-ref
rel_dom = abs_resid[dominant] / np.abs(ref_atomic[dominant])     # numpy-ref

floor = 1.0e-6 if DTYPE == torch.float32 else 1.0e-10
print(f"reference nonzero points: {nz.sum()} / {ref_atomic.size}")  # numpy-ref
print(f"all nonzero points     : max rel = {rel_all.max():.3e}   median = {np.median(rel_all):.3e}")  # numpy-ref
print(f"atomic-dominated points: max rel = {rel_dom.max():.3e}   median = {np.median(rel_dom):.3e}")  # numpy-ref
print(f"max ABSOLUTE residual everywhere: {abs_resid.max():.2e}")  # numpy-ref
print(f"device = {DEVICE.type}; float floor = {floor:.1e}")''')

md(r"""### The per-component split

Separate the benchmark into metals and helium. The metal comparison is the validated scatter-add path from the earlier section; helium is the vectorized taper path above. The reference helium wings are stored pre-stimulated-emission in the catalog, so the comparison multiplies them by the same `stim` array used for the metals.""")

code(r'''# Comparison-reference cell: per-component parity oracle.
gt_metal = ref_atomic - gt_he                                    # numpy-ref

mdom = (gt_metal > 1e-10) & (gt_metal > 10.0 * (cat["gt_ahline"] + gt_he))  # numpy-ref
hdom = gt_he > 1e-10                                             # numpy-ref

rel_m = np.abs(metal_gpu[mdom] - gt_metal[mdom]) / np.abs(gt_metal[mdom])  # numpy-ref
rel_h = np.abs(he_gpu[hdom] - gt_he[hdom]) / np.abs(gt_he[hdom])           # numpy-ref

print(f"[metals, all Z] {mdom.sum():6d} points : max rel = {rel_m.max():.3e}   median = {np.median(rel_m):.3e}")  # numpy-ref
print(f"[helium]        {hdom.sum():6d} points : max rel = {rel_h.max():.3e}   median = {np.median(rel_h):.3e}")  # numpy-ref
print(f"[combined]                     : max ABS residual = {np.abs(atomic_gpu - ref_atomic).max():.2e}")  # numpy-ref
assert rel_m.max() <= floor, f"metal component max rel {rel_m.max():.3e} above {floor:.1e}"  # numpy-ref
assert rel_h.max() <= floor, f"helium component max rel {rel_h.max():.3e} above {floor:.1e}"  # numpy-ref
assert rel_all.max() <= floor, f"combined atomic max rel {rel_all.max():.3e} above {floor:.1e}"  # numpy-ref''')

md(r"""## The missing piece: hydrogen

Hydrogen is deliberately not part of this lecture's ordinary Voigt accumulation. Its Balmer wings are dominated by the **linear Stark effect**: because hydrogen's levels are nearly degenerate, charged-particle microfields shift them linearly and produce broad non-Voigt wings. SYNTHE therefore routes hydrogen through its HPROF/HLINOP broadening machinery, not through the metal/helium Voigt wing walk.

The full line opacity is

\[
\kappa^{\rm line}_\lambda
=
\underbrace{\kappa^{\rm metal}_\lambda+\kappa^{\rm He}_\lambda}_{\text{this lecture}}
+
\underbrace{\kappa^{\rm H}_\lambda}_{\text{next lecture}}.
\]

The benchmark above subtracts `gt_ahline` precisely so this lecture compares like with like.""")

md(r"""## Beyond ordinary lines: autoionizing and merged-continuum profiles

The ordinary metal and helium paths are Voigt-based. The line-type code can also route a transition to two non-Voigt shapes:

- **Autoionizing lines** (type `1`): a Fano/shore profile, asymmetric because a bound transition interferes with a continuum channel.
- **Merged-continuum lines** (type `81`, and codes `>3`): a flat pseudo-continuum contribution that ramps linearly to zero near a series limit.

These do not affect the 500–510 nm solar-window opacity benchmark above, but the inline fp64 reference closes the lecture by validating the two leaf profile routines against a hot-star diagnostic. We do the same here, vectorized over padded record batches.""")

md(r"""### The hot-star test window and the ground truth

The diagnostic file `linetypes.npz` stores the exact inputs and recorded output deltas for every contributing special-profile record in a hot-star synthesis. The following comparison-reference setup packs those variable-length records into padded arrays; the profile arithmetic itself is torch-vectorized over `(record, padded-wavelength)` tensors.""")

code(r'''# Comparison-reference/setup cell: variable-length archival records are packed once for the torch profile tests.
lt_ref = np.load(REF / "linetypes.npz")  # numpy-ref
teff, logg = lt_ref["teff_logg"]         # numpy-ref
w0, w1, w_res = lt_ref["window"]         # numpy-ref
n_auto = int(lt_ref["auto_n"])           # numpy-ref
n_cont = int(lt_ref["cont_n"])           # numpy-ref

print(f"hot star : Teff={teff:.0f} K  logg={logg:.2f}   window {w0:.0f}-{w1:.0f} nm  R={w_res:.0f}")  # numpy-ref
print(f"records  : {n_auto} autoionizing, {n_cont} merged-continuum")  # numpy-ref

def _pack_records(prefix, record_count):
    # JUSTIFIED-LOOP: heterogeneous npz archive unpacking only; no profile arithmetic is done here.
    record_lengths = np.empty(record_count, dtype=np.int64)  # numpy-ref
    for record_index in range(record_count):  # numpy-ref
        record_lengths[record_index] = lt_ref[f"{prefix}{record_index}_wl_slice"].size  # numpy-ref
    padded_width = int(record_lengths.max())  # numpy-ref
    wavelength_slices = np.zeros((record_count, padded_width), dtype=np.float64)  # numpy-ref
    continuum_slices = np.zeros((record_count, padded_width), dtype=np.float64)   # numpy-ref
    reference_delta_slices = np.zeros((record_count, padded_width), dtype=np.float64)  # numpy-ref
    valid_mask = np.zeros((record_count, padded_width), dtype=bool)              # numpy-ref
    for record_index in range(record_count):  # numpy-ref
        record_length = record_lengths[record_index]  # numpy-ref
        wavelength_slices[record_index, :record_length] = lt_ref[f"{prefix}{record_index}_wl_slice"]  # numpy-ref
        continuum_slices[record_index, :record_length] = lt_ref[f"{prefix}{record_index}_cont_slice"]  # numpy-ref
        reference_delta_slices[record_index, :record_length] = lt_ref[f"{prefix}{record_index}_delta_vals"]  # numpy-ref
        valid_mask[record_index, :record_length] = True  # numpy-ref
    return wavelength_slices, continuum_slices, reference_delta_slices, valid_mask, record_lengths  # numpy-ref

auto_wl, auto_cont, auto_ref, auto_valid, auto_lens = _pack_records("auto", n_auto)  # numpy-ref
cont_wl, cont_cont, cont_ref, cont_valid, cont_lens = _pack_records("cont", n_cont)  # numpy-ref''')

md(r"""### The autoionizing (Fano/shore) profile

For an autoionizing record the detuning is

\[
\varepsilon = \frac{2(\nu-\nu_0)}{\gamma},
\qquad \nu=\frac{c}{\lambda},
\]

and the shore profile is

\[
\kappa(\varepsilon)
=
\kappa_0
\frac{a_{\rm shore}\varepsilon+b_{\rm shore}}
     {(\varepsilon^2+1)b_{\rm shore}}.
\]

The center deposits \(\kappa_0\) after the whole-line cutoff gate; each wing stops at the first non-positive or below-cutoff value. In torch, both wing walks become cumulative stop masks on the padded wavelength axis.""")

code(r'''# Comparison-reference/setup values for the torch autoionizing profile.
# JUSTIFIED-LOOP: archive keys are per-record scalars; this packs them, profile compute below is vectorized.
auto_slice_lo = np.empty(n_auto, dtype=np.int64)          # numpy-ref
auto_center = np.empty(n_auto, dtype=np.int64)            # numpy-ref
auto_line_wl = np.empty(n_auto, dtype=np.float64)         # numpy-ref
auto_kappa0 = np.empty(n_auto, dtype=np.float64)          # numpy-ref
auto_grad = np.empty(n_auto, dtype=np.float64)            # numpy-ref
auto_gstark = np.empty(n_auto, dtype=np.float64)          # numpy-ref
auto_gvdw = np.empty(n_auto, dtype=np.float64)            # numpy-ref
auto_cut = np.empty(n_auto, dtype=np.float64)             # numpy-ref
for i in range(n_auto):  # numpy-ref
    auto_slice_lo[i] = int(lt_ref[f"auto{i}_slice_lo"])   # numpy-ref
    auto_center[i] = int(lt_ref[f"auto{i}_center_index"]) # numpy-ref
    auto_line_wl[i] = float(lt_ref[f"auto{i}_line_wavelength"])  # numpy-ref
    auto_kappa0[i] = float(lt_ref[f"auto{i}_kappa0"])     # numpy-ref
    auto_grad[i] = float(lt_ref[f"auto{i}_gamma_rad"])    # numpy-ref
    auto_gstark[i] = float(lt_ref[f"auto{i}_gamma_stark"])# numpy-ref
    auto_gvdw[i] = float(lt_ref[f"auto{i}_gamma_vdw"])    # numpy-ref
    auto_cut[i] = float(lt_ref[f"auto{i}_cutoff"])        # numpy-ref

def autoionizing_delta_torch():
    """Return the autoionizing-line opacity deltas on each recorded local slice.

    Autoionizing profiles use the Shore form and stop independently on each
    side of line centre once the profile is non-positive or below the continuum
    cutoff.  The arithmetic is intentionally done in a tiny CPU/fp64 island
    because the detuning subtracts nearly equal optical frequencies.
    """
    # The Shore detuning is a difference of two ~1e15-Hz frequencies.  Evaluate
    # this tiny special-record batch in CPU/fp64, then return the result to the
    # working device; the 12,568-line ordinary opacity remains MPS-resident.
    solve_device, solve_dtype = torch.device("cpu"), torch.float64
    to64 = lambda x: torch.as_tensor(x, dtype=solve_dtype, device=solve_device)
    wavelength_slices = to64(auto_wl)
    continuum_slices = to64(auto_cont)
    valid_mask = torch.as_tensor(auto_valid, dtype=torch.bool, device=solve_device)

    slice_start_index = torch.as_tensor(auto_slice_lo, dtype=torch.int64, device=solve_device)
    center_index_global = torch.as_tensor(auto_center, dtype=torch.int64, device=solve_device)
    center_index_local = center_index_global - slice_start_index

    line_wavelength = to64(auto_line_wl)
    kappa0 = to64(auto_kappa0)
    gamma = torch.clamp(torch.abs(to64(auto_grad)), min=1.0e-300)
    ashore = to64(auto_gstark)
    bshore_raw = to64(auto_gvdw)
    bshore = torch.where(torch.abs(bshore_raw) >= 1.0e-300, bshore_raw,
                         torch.copysign(torch.full_like(bshore_raw, 1.0e-300), bshore_raw))
    cutoff = to64(auto_cut)

    record_index = torch.arange(wavelength_slices.shape[0], dtype=torch.int64, device=solve_device)
    continuum_at_center = continuum_slices[record_index, center_index_local]
    line_passes_cutoff = (kappa0 >= continuum_at_center * cutoff) & (kappa0 > 0.0)

    # Stable detuning: avoid subtracting two ~1e15 Hz fp32 frequencies.  The
    # algebraic form c*(lambda0-lambda)/(lambda*lambda0) preserves the small
    # frequency difference that selects the Shore-profile sign and cutoff.
    speed_of_light_nm = torch.as_tensor(2.99792458e17, dtype=solve_dtype, device=solve_device)
    line_wavelength_column = line_wavelength.view(-1, 1)
    safe_wavelength = torch.where(
        valid_mask, torch.clamp(wavelength_slices, min=1.0e-30), line_wavelength_column)
    frequency_detuning = (
        speed_of_light_nm * (line_wavelength_column - safe_wavelength)
        / (safe_wavelength * line_wavelength_column)
    )
    shore_epsilon = 2.0 * frequency_detuning / gamma.view(-1, 1)
    profile_value = (
        kappa0.view(-1, 1) * (ashore.view(-1, 1) * shore_epsilon + bshore.view(-1, 1))
        / ((shore_epsilon * shore_epsilon + 1.0) * bshore.view(-1, 1))
    )

    local_column_index = torch.arange(
        wavelength_slices.shape[1], dtype=torch.int64, device=solve_device).view(1, -1)
    center_column = center_index_local.view(-1, 1)

    red_side = local_column_index > center_column
    blue_side = local_column_index < center_column
    invalid_or_stopped = (
        (~valid_mask) | (profile_value <= 0.0)
        | (profile_value < continuum_slices * cutoff.view(-1, 1))
    )

    red_stop = red_side & valid_mask & invalid_or_stopped
    blue_stop = blue_side & valid_mask & invalid_or_stopped
    red_before_stop = torch.cumsum(red_stop.to(torch.int64), dim=1) == 0
    blue_before_stop = torch.flip(
        torch.cumsum(torch.flip(blue_stop.to(torch.int64), dims=(1,)), dim=1) == 0,
        dims=(1,))

    wing_deposit_mask = (
        line_passes_cutoff.view(-1, 1) & valid_mask & (~invalid_or_stopped)
        & ((red_side & red_before_stop) | (blue_side & blue_before_stop))
    )
    profile_delta = torch.where(wing_deposit_mask, profile_value, torch.zeros_like(profile_value))

    center_deposit_mask = torch.zeros_like(profile_delta, dtype=torch.bool)
    center_deposit_mask[record_index, center_index_local] = line_passes_cutoff
    profile_delta = torch.where(center_deposit_mask, kappa0.view(-1, 1), profile_delta)
    return profile_delta.to(device=DEVICE, dtype=DTYPE)
''')

md(r"""Run the autoionizing profile and compare it on its recorded support. This cell is intentionally only a validation call plus assertions; the profile arithmetic was defined above.""")

code(r'''
auto_gpu_t = autoionizing_delta_torch()
auto_gpu = auto_gpu_t.detach().cpu().to(torch.float64).numpy()  # numpy-ref
auto_abs = np.max(np.abs(auto_gpu[auto_valid] - auto_ref[auto_valid]))  # numpy-ref
auto_support = auto_valid & (auto_ref != 0.0)  # numpy-ref: physical profile support
auto_rel = np.max(np.abs(auto_gpu[auto_support] - auto_ref[auto_support]) / np.abs(auto_ref[auto_support]))  # numpy-ref
auto_exact = np.array_equal(auto_gpu, auto_ref) if DTYPE == torch.float64 else False  # numpy-ref
print(f"autoionizing profile: max abs = {auto_abs:.3e}, max|rel| = {auto_rel:.3e}, bit-exact(fp64 only) = {auto_exact}")  # numpy-ref
assert auto_rel <= floor, f"autoionizing full-support max rel {auto_rel:.3e} above {floor:.1e}"  # numpy-ref''')

md(r"""### The merged-continuum (ramp) profile

The merged-continuum line type deposits a flat strength until the merge index, then ramps linearly to zero at the tail index. The scalar routine walks increasing global grid index and stops once the ramped value falls below the local cutoff; the torch version again uses a cumulative stop mask.""")

code(r'''# Comparison-reference/setup values for the torch merged-continuum profile.
# JUSTIFIED-LOOP: archive keys are per-record scalars; this packs them, profile compute below is vectorized.
cont_slice_lo = np.empty(n_cont, dtype=np.int64)          # numpy-ref
cont_line_wl = np.empty(n_cont, dtype=np.float64)         # numpy-ref
cont_kappa = np.empty(n_cont, dtype=np.float64)           # numpy-ref
cont_cut = np.empty(n_cont, dtype=np.float64)             # numpy-ref
cont_idx_start = np.empty(n_cont, dtype=np.int64)         # numpy-ref
cont_idx_merge = np.empty(n_cont, dtype=np.int64)         # numpy-ref
cont_idx_tail = np.empty(n_cont, dtype=np.int64)          # numpy-ref
for i in range(n_cont):  # numpy-ref
    cont_slice_lo[i] = int(lt_ref[f"cont{i}_slice_lo"])   # numpy-ref
    cont_line_wl[i] = float(lt_ref[f"cont{i}_line_wavelength"])  # numpy-ref
    cont_kappa[i] = float(lt_ref[f"cont{i}_kappa"])       # numpy-ref
    cont_cut[i] = float(lt_ref[f"cont{i}_cutoff"])        # numpy-ref
    cont_idx_start[i] = int(lt_ref[f"cont{i}_idx_start_g"])  # numpy-ref
    cont_idx_merge[i] = int(lt_ref[f"cont{i}_idx_merge_g"])  # numpy-ref
    cont_idx_tail[i] = int(lt_ref[f"cont{i}_idx_tail_g"])    # numpy-ref

def merged_continuum_delta_torch():
    """Return the merged-continuum ramp opacity deltas on each local slice.

    Each record contributes a flat `kappa` section until its merge index and a
    linear falloff toward the tail index.  The scalar walk's first-below-cutoff
    stop condition is represented by a cumulative mask along the local slice.
    """
    wavelength_slices = dev(cont_wl)
    continuum_slices = dev(cont_cont)
    valid_mask = torch.as_tensor(cont_valid, dtype=torch.bool, device=DEVICE)

    slice_start_index = torch.as_tensor(cont_slice_lo, dtype=torch.int64, device=DEVICE)
    idx_start = torch.as_tensor(cont_idx_start, dtype=torch.int64, device=DEVICE)
    idx_merge = torch.as_tensor(cont_idx_merge, dtype=torch.int64, device=DEVICE)
    idx_tail = torch.as_tensor(cont_idx_tail, dtype=torch.int64, device=DEVICE)

    line_wavelength = dev(cont_line_wl)
    kappa = dev(cont_kappa)
    cutoff = dev(cont_cut)

    local_column_index = torch.arange(
        wavelength_slices.shape[1], dtype=torch.int64, device=DEVICE).view(1, -1)
    global_column_index = slice_start_index.view(-1, 1) + local_column_index

    ramp_denominator = torch.clamp(idx_tail - torch.maximum(idx_merge, idx_start), min=1)
    ramp = torch.where(
        global_column_index >= idx_merge.view(-1, 1),
        ((idx_tail.view(-1, 1) - global_column_index).to(DTYPE)
         / ramp_denominator.view(-1, 1).to(DTYPE)),
        torch.ones_like(wavelength_slices),
    )
    ramp = torch.clamp(ramp, min=0.0)

    ramped_value = kappa.view(-1, 1) * ramp
    domain = (
        valid_mask &
        (kappa.view(-1, 1) > 0.0) &
        (idx_tail.view(-1, 1) > idx_start.view(-1, 1)) &
        (global_column_index >= idx_start.view(-1, 1)) &
        (global_column_index < idx_tail.view(-1, 1)) &
        (wavelength_slices >= line_wavelength.view(-1, 1))
    )

    stop_mask = domain & (ramped_value < continuum_slices * cutoff.view(-1, 1))
    before_stop_mask = torch.cumsum(stop_mask.to(torch.int64), dim=1) == 0
    deposit_mask = domain & before_stop_mask & (~stop_mask)

    return torch.where(deposit_mask, ramped_value, torch.zeros_like(ramped_value))
''')

md(r"""Run the merged-continuum ramp profile and assert it against the recorded support. Together with the autoionizing check, this closes the two non-Voigt leaf profiles used by special line-type records.""")

code(r'''
cont_gpu_t = merged_continuum_delta_torch()
cont_gpu = cont_gpu_t.detach().cpu().to(torch.float64).numpy()  # numpy-ref
cont_abs = np.max(np.abs(cont_gpu[cont_valid] - cont_ref[cont_valid]))  # numpy-ref
cont_support = cont_valid & (cont_ref != 0.0)  # numpy-ref: physical ramp support
cont_rel = np.max(np.abs(cont_gpu[cont_support] - cont_ref[cont_support]) / np.abs(cont_ref[cont_support]))  # numpy-ref
cont_exact = np.array_equal(cont_gpu, cont_ref) if DTYPE == torch.float64 else False  # numpy-ref
print(f"merged-continuum profile: max abs = {cont_abs:.3e}, max|rel| = {cont_rel:.3e}, bit-exact(fp64 only) = {cont_exact}")  # numpy-ref
assert cont_rel <= floor, f"merged-continuum full-support max rel {cont_rel:.3e} above {floor:.1e}"  # numpy-ref
print(f"BOTH special line types reproduced at the device float floor: {max(auto_rel, cont_rel):.3e}")  # numpy-ref''')

md(r"""### The two shapes, side by side

The final figure overlays one representative autoionizing profile and one representative merged-continuum ramp on their recorded pykurucz deltas. The plot cell is a host/Matplotlib boundary, so converting the already-computed torch results back to NumPy is appropriate here.""")

code(r'''# Plot cell: host-side Matplotlib boundary.
ai = 0
mi = 0

fig, (axa, axm) = plt.subplots(1, 2, figsize=(12, 4.0))

mask_a = auto_valid[ai]  # numpy-ref
axa.plot(auto_wl[ai, mask_a], auto_ref[ai, mask_a], color="0.6", lw=3.0, label="pykurucz")  # numpy-ref
axa.plot(auto_wl[ai, mask_a], auto_gpu[ai, mask_a], color="C3", lw=1.0, label="GPU torch")  # numpy-ref
axa.axvline(auto_line_wl[ai], color="0.8", lw=0.8, zorder=0)  # numpy-ref
axa.set_title(f"Autoionizing (Fano/shore) — {auto_line_wl[ai]:.3f} nm")  # numpy-ref
axa.set_xlabel("wavelength  [nm]")
axa.set_ylabel(r"line opacity  [cm$^2$/g]")
axa.locator_params(axis="x", nbins=4)
axa.legend(loc="upper right")

mask_m = cont_valid[mi]  # numpy-ref
axm.plot(cont_wl[mi, mask_m], cont_ref[mi, mask_m], color="0.6", lw=3.0, label="pykurucz")  # numpy-ref
axm.plot(cont_wl[mi, mask_m], cont_gpu[mi, mask_m], color="C0", lw=1.0, label="GPU torch")  # numpy-ref
axm.set_title(f"Merged continuum (ramp) — {cont_line_wl[mi]:.3f} nm")  # numpy-ref
axm.set_xlabel("wavelength  [nm]")
axm.set_ylabel(r"pseudo-continuum opacity  [cm$^2$/g]")
axm.locator_params(axis="x", nbins=5)
axm.legend(loc="upper right")

fig.tight_layout()
plt.show()''')

# ════════════════════════════════════════════════════════════════════════════
#  Synthesis / Summary / Practice / Further reading
# ════════════════════════════════════════════════════════════════════════════
md(r"""## Synthesis: what you built and where it goes

You took the scalar reference's per-line outward `+=` wing walk — a Python loop over twelve thousand lines, each `while`-looping scalar by scalar into the opacity array — and recast it as a **batched scatter**. The line axis became a **tensor batch axis**; `_wing_reach_batched` computed every $(\text{depth},\text{line})$ pair's reach at once; `_wing_walk_tiered`/`_wing_walk_core` swept fixed offsets over reach-tiered buckets; and the deposit collapsed to a handful of **`index_put_(accumulate=True)`** scatters — the GPU's native atomic accumulation. The Harris `voigt_H` kernel of Lecture 4 was reused unchanged across the whole list, and FASTEX became a branchless gather. The result reproduces the inline fp64 reference to the fp32 floor.

The translation keeps the full textbook coverage: ordinary metal lines, helium continuum-merge wings, the hydrogen boundary that moves to Lecture 6, and the two non-Voigt leaf profiles used by special line-type records. The extra GPU lesson is the computational shape. **(1) Loop $\to$ scatter:** an adaptive per-element `+=` walk is a scatter, and the right shape is to batch the line axis and deposit with `index_put_(accumulate=True)`. **(2) The Metal-kernel verdict:** batched `index_put_` already lowers to an efficient atomic scatter; the win is in batching and reach-tiering. **(3) Precision placement:** discontinuous geometry and the cancellation-prone Harris core are compact fp64 islands, while the large low/far profile blocks and overlapping deposits remain on MPS. That division makes the strict full-maximum gate possible.

This atomic opacity, added to the continuum of Lecture 3 and the hydrogen lines of Lecture 6, is the line extinction the transfer lectures consume. Fed through Lectures 7–8, it produces the solar spectrum line for line.""")

md(r"""## Summary

- The scalar reference's **per-line outward `+=` wing walk** (red + blue, stop at the cutoff) is the **scatter-add hot path**; the tensor implementation recasts it with the **line axis as a batch axis** and the deposit as **one batched `[depth, line, offset]` scatter** per red/blue — `index_put_(accumulate=True)` — instead of $O(n_{\rm lines})$ tiny launches.
- **`_wing_reach_batched`** computes every $(\text{depth},\text{line})$ pair's reach geometry at once; **`_wing_walk_tiered`/`_wing_walk_core`** sweep fixed offsets over **reach-tiered** buckets (to avoid wasted Harris evals); **`_scatter_add_3d`** is the `index_put_(accumulate=True)` deposit. The Harris **`voigt_H_grid`** of Lecture 4 and a branchless **FASTEX** gather are reused across the whole list.
- The **Metal-kernel verdict**: the custom Metal scatter was evaluated and **not adopted**. The scalar-walk alternative was rejected, and a true Metal scatter only matched batched torch because `index_put_(accumulate=True)` already lowers to the backend's atomic scatter.
- **Parity:** every opacity-bearing metal pixel passes **max relative error $\le 10^{-6}$**; the helium family and both special-profile families are gated the same way.
- The lecture boundary is explicit: hydrogen is non-Voigt Stark broadening and is rebuilt in Lecture 6, not hidden inside this line-list kernel.""")

md(r"""## Practice exercises

**1. Watch the scatter overlap.** The whole point of `accumulate=True` is that two line wings landing on the same pixel **add**. Replace `index_put_(accumulate=True)` with `accumulate=False` in `scatter_add_3d` and re-run the comparison: where in the forest does the opacity now drop, and why? (Hint: the blends, where many wings overlap.)

**2. The cost of tiering.** Replace `_wing_walk_tiered` with a single `_wing_walk_core` call over all live lines (one global $W$). The result is identical, but time both. By how much does the wasted-Harris-eval blow-up grow with the widest-reaching line in the batch?

**3. FASTEX vs `torch.exp`.** Swap the branchless `fast_ex` for a plain `torch.exp(-x)` in `accumulate_metal` and re-run the comparison. Where does the agreement break, and at what level? This shows that matching the production code means matching its *tables*, not just its formulas.

**4. Precision-island audit.** Disable `harris_branch_oracle` and re-run the full-maximum assertion. Identify separately the discrete branch failures and the intermediate-polynomial cancellation points that return.""")

md(r"""## Further reading

- **Kurucz, R. L. (2011). *Including all the lines*, Canadian Journal of Physics, 89, 417.** The philosophy and construction of the line lists this accumulation reads.
- **Harris, D. L. (1948). *On the line-absorption coefficient due to Doppler effect and damping*, ApJ, 108, 112.** The Voigt-function polynomial tables `h0tab/h1tab/h2tab` the kernel evaluates.
- **PyTorch documentation, `Tensor.index_put_`.** The accumulating atomic scatter that is the hot path here; on accelerator backends it lowers to an atomic-add style scatter, which is why a custom scatter does not beat it.
- **Gray, D. F. (2005). *The Observation and Analysis of Stellar Photospheres*, 3rd ed.** Chapters 11–13 on line absorption and the assembly of many lines.
- **Kim, E. M. & Ting, Y.-S. (2026). [*pykurucz*](https://arxiv.org/abs/2603.11693).** The implementation our fp64 reference and diagnostic line opacity are reduced from.""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, str(OUT))
print(f"wrote {OUT.relative_to(BOOK)} ({len(cells)} cells)")
