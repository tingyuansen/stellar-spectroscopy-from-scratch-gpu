# Chapter 9 first-pass contract — Radiative Transfer with Scattering

Status: bounded reader-facing design; no implementation or publication authority  
Pinned Payne Zero commit: `9c44001feae40b85146630499e6f8a5fed42e5af`  
Audience: final-year undergraduate / first-year graduate student  
Canonical title: **Radiative Transfer with Scattering**

## 0. Canonical placement and ownership

This chapter begins only after the reader can construct the thermochemical
state and the continuous, atomic, and molecular opacity contributions:

```text
Chapters 3–4: particles available at every depth
Chapters 5–8: absorption and scattering at every depth and wavelength
                    |
                    v
Chapter 9: radiation that actually escapes
                    |
                    v
Chapter 10: reusable window state and the complete device pipeline
```

The chapter owns the physical and numerical transfer calculation. It does not
rederive any opacity process, line profile, catalog route, or molecular
compiler. It also does not build the Chapter 10 cache/device orchestration.

Chapter 2 already derived and checked the parabolic optical-depth integrators,
the ordered depth prefix, and the safe Numba pattern of independent frequency
chunks with private accumulators. Chapter 9 may recall those results in one
compact bridge, then must use them to answer a new physical question. It must
not repeat the Chapter 2 tutorial on `njit`, `prange`, array strides, or timing.

Chapter 9 teaches the common transfer physics once and then makes the two
production lanes explicit:

- the synthesis lane needs emergent total and continuum Eddington flux
  \(H_\nu\) for a wavelength batch;
- the atmosphere lane needs \(J_\nu\), \(H_\nu\), \(J_\nu-S_\nu\), and a
  surface second moment at every sampled frequency and depth so later
  chapters can form frequency-integrated atmosphere corrections.

The single-frequency atmosphere moment solve belongs here because it is
radiative transfer. Chapter 12 owns the *lifecycle and physical use* of the
30,000-frequency accumulators—Rosseland opacity, radiation pressure,
radiative acceleration, heating, and temperature-correction integrals. This
division prevents Chapter 12 from re-teaching transfer while keeping Chapter
9 from becoming an atmosphere-iteration chapter.

There are no detached exercises. Every limiting case or numerical comparison
appears where it resolves a live question and is interpreted immediately.

## 1. The chapter's single question

Open with two depth columns having the same total extinction
\(\chi_\nu\), the same temperature profile, and therefore the same optical
depth, but different divisions between true absorption and scattering. One
column thermalizes efficiently; the other redirects photons and depends on
the radiation arriving from other layers.

Ask:

> Given depth-dependent absorption, emission, and scattering, what total and
> continuum flux actually leave the stellar surface?

This opening exposes the missing information in an extinction slab. Optical
depth says how difficult escape is, but not what replaces radiation removed
from a ray. The chapter's causal construction is

```text
absorption + scattering + thermal emissivity
    -> extinction, scattering fraction, and thermal source
    -> monochromatic optical depth
    -> source on a common 51-point optical-depth grid
    -> scattering source iteration
    -> emergent H_nu
    -> total and continuum branches
    -> F_nu, F_lambda per nm, and normalized flux
```

The first numerical prediction should be stated before code:

- if scattering is zero, the source is local and thermal;
- if line opacity is zero, total and continuum transfer must be identical;
- if two columns have the same extinction but different scattering fractions,
  their emergent flux need not be the same.

The chapter should answer those three predictions progressively, not display
them as a final checklist.

## 2. Reader promise, assumptions, and honest scope

By the end of the chapter, the reader should be able to explain and reproduce:

- why total extinction is not enough to determine a source function;
- how \(d\tau_\nu=\chi_\nu\,dm\) turns the physical depth grid into a
  wavelength-specific optical-depth grid;
- what specific intensity and its moments \(J_\nu\), \(H_\nu\), and
  \(K_\nu\) measure;
- how angular quadrature turns ray intensities into moments without confusing
  \(H_\nu\) with the physical flux \(F_\nu\);
- why the scattering source contains the unknown mean intensity;
- why all wavelengths may be batched while the backward depth sweep remains
  ordered;
- what the 51-point operator table represents and why it is static physical
  input rather than a golden answer;
- why the synthesis solver uses exactly eight float32 source sweeps;
- why the atmosphere moment lane has a different deep-layer continuation and
  convergence policy;
- how total and continuum branches are solved together;
- when the saturated-core fallback is required;
- how \(H_\nu\), \(F_\nu\), and \(F_\lambda\) differ, including the factor
  \(4\pi\) and the frequency-to-wavelength Jacobian;
- why `normalized_flux` is a ratio of two physical flux calculations, not an
  arbitrary continuum fit.

Assume only the book's basic calculus, arrays, and Chapter 1 meanings of
optical depth and a source function. Define at first use:

- a **ray** as radiation travelling in one direction;
- the direction cosine \(\mu=\cos\theta\), with \(\theta\) measured from the
  outward surface normal;
- an **angular moment** as a weighted average over ray direction;
- a **formal solution** as the exact integral solution for a known source;
- a **Lambda operator** as the linear mapping from a complete source vector
  to the resulting mean-intensity vector;
- a **source iteration** as repeated reconciliation of a source with the
  radiation field it creates;
- a **boundary condition** as the supplied behavior of rays entering the
  top or deep edge of the computational domain.

The supported physical model remains one-dimensional, static,
plane-parallel, and LTE for material populations and thermal emissivity.
Standard line opacity is thermal absorption. Standard synthesis supplies
`line_scattering = 0`; continuum scattering is the active non-thermal source
term.

This chapter does **not** implement:

- non-LTE statistical equilibrium or departure-coefficient iteration;
- partial frequency redistribution, even when a catalog record carried a
  `PRD` tag in Chapter 7;
- frequency-coupled scattering, polarization, time dependence, spherical
  geometry, winds, or three-dimensional transfer;
- an instrumental line-spread function, observed-pixel sampling, fitting, or
  continuum placement.

The scattering source iteration updates the radiation field while holding
temperature, populations, and opacity fixed. It is therefore not an NLTE
population solve. State this at the moment source iteration first appears.

## 3. One notation ledger, then exact array names

Use frequency notation internally because the production transfer returns
\(H_\nu\). Wavelength is the public spectral coordinate. Do not switch
silently between \(B_\nu\) and \(B_\lambda\).

| symbol | physical meaning | exact executable name | unit |
| --- | --- | --- | --- |
| \(m\) | column mass, increasing inward | `column_mass` | g cm\(^{-2}\) |
| \(\kappa_{\nu,\mathrm{c}}\) | continuum true absorption per mass | `continuum_absorption` | cm\(^2\) g\(^{-1}\) |
| \(\sigma_{\nu,\mathrm{c}}\) | continuum scattering per mass | `continuum_scattering` | cm\(^2\) g\(^{-1}\) |
| \(\kappa_{\nu,\mathrm{l}}\) | line true absorption per mass | `line_mass_absorption_coefficient` | cm\(^2\) g\(^{-1}\) |
| \(\sigma_{\nu,\mathrm{l}}\) | line scattering per mass | `line_scattering` | cm\(^2\) g\(^{-1}\) |
| \(\chi_\nu\) | total extinction per mass | `total_extinction` or local `extinction` | cm\(^2\) g\(^{-1}\) |
| \(\alpha_\nu\) | scattering fraction \((\sigma_{\rm c}+\sigma_{\rm l})/\chi_\nu\) | `scattering_fraction` | dimensionless |
| \(S_{\nu,\mathrm{th}}\) | absorption-weighted thermal source | `thermal_source` | same spectral-radiance unit as \(B_\nu\) |
| \(S_\nu\) | complete thermal-plus-scattered source | returned `source` | same spectral-radiance unit |
| \(\tau_\nu\) | inward optical depth | `optical_depth` | dimensionless |
| \(J_\nu\) | mean-intensity moment | `mean_intensity` | same unit as \(I_\nu\) |
| \(H_\nu\) | Eddington-flux moment | `eddington_flux_*_per_frequency` | flux-per-frequency divided by \(4\pi\) |
| \(K_\nu\) | second angular moment | represented by the atmosphere `surface_second_moment` | same unit as \(I_\nu\) |
| \(F_\nu\) | surface flux per frequency | \(4\pi H_\nu\) | erg s\(^{-1}\) cm\(^{-2}\) Hz\(^{-1}\) |
| \(F_\lambda\) | surface flux per wavelength | `Spectrum.flux_total`, `Spectrum.flux_continuum` | erg s\(^{-1}\) cm\(^{-2}\) nm\(^{-1}\) |
| \(f_{\rm norm}\) | total-to-continuum ratio | `normalized_flux` | dimensionless |

Avoid using \(\epsilon_\nu\) without a qualifier. The paper uses
\(\epsilon_\nu=1-\alpha_\nu\) for the thermal/absorption fraction, while the
exact function returns `scattering_fraction = alpha`. The executable source
equation is

\[
S_\nu=(1-\alpha_\nu)S_{\nu,\mathrm{th}}+\alpha_\nu J_\nu .
\]

Writing \(S=\epsilon B+(1-\epsilon)J\) is acceptable only after explicitly
defining \(\epsilon=1-\alpha\). The notebook should use `scattering_fraction`
in code so the complement cannot be silently reversed.

## 4. Exact reads, writes, axes, units, dtype, and device

Let \(D\) be physical depth layers, \(W\) wavelength samples, and \(T=51\) the
fixed transfer-grid length.

### 4.1 Synthesis transfer boundary

The public pipeline supplies depth-major slabs:

| exact input | shape on entry | unit | normal standard value |
| --- | ---: | --- | --- |
| `continuum_absorption` | `(D,W)` | cm\(^2\) g\(^{-1}\) | Chapter 5 result |
| `continuum_scattering` | `(D,W)` | cm\(^2\) g\(^{-1}\) | Chapter 5 result |
| `line_mass_absorption_coefficient` | `(D,W)` | cm\(^2\) g\(^{-1}\) | Chapters 6–8 sum |
| `line_scattering` | `(D,W)` | cm\(^2\) g\(^{-1}\) | zero in standard LTE synthesis |
| `planck_source` | `(D,W)` | internal \(B_\nu\) scale | `planck_bnu(...)` |
| `column_mass` | `(D,)` | g cm\(^{-2}\) | strictly increasing inward |
| `tables` | one `TransferTables` | static operators | selected device/work dtype |

`solve_spectrum(...)` transposes the slabs contiguously so wavelength is the
batch axis and depth is ordered:

```text
entry:     (depth, wavelength)
transfer:  (wavelength, depth)
grid:      (wavelength, 51)
output:    (wavelength,)
```

It returns, on the selected Torch device:

- `eddington_flux_total_per_frequency (W,)`;
- `eddington_flux_continuum_per_frequency (W,)`;
- `normalized_flux (W,)`.

The total and continuum \(H_\nu\) arrays are not yet `Spectrum.flux_total` and
`Spectrum.flux_continuum`; the latter are public \(F_\lambda\) per nm after
the factor \(4\pi\) and Jacobian. Chapter 9 derives and checks that conversion.
Chapter 10 still owns the one bulk host crossing in the complete pipeline.

### 4.2 Atmosphere single-frequency transfer boundary

The atmosphere lane consumes one frequency column at a time:

| exact input family | shape | working dtype/meaning |
| --- | ---: | --- |
| `continuum_absorption`, `continuum_scattering`, `continuum_source` | `(D,)` | NumPy float64 |
| gross `line_mass_absorption_coefficient` | `(D,)` | stored float32, converted and multiplied by the float64 `stimulated` factor at transfer |
| `column_mass`, `planck` | `(D,)` | NumPy float64 |
| `transfer_grid` | `(51,)` | float64 |
| `mean_intensity_operator`, `eddington_flux_operator`, `second_moment_weights` | `(51,51)`, `(51,51)`, `(51,)` | loaded float32 |

The standard atmosphere route has no line-scattering term. Its exact total
opacity is continuum absorption plus stimulated line absorption plus
continuum scattering.

The one-frequency kernel writes physical-depth arrays:

- `optical_depth_out (D,)`;
- `source_out (D,)`;
- `eddington_flux_out (D,)`;
- `mean_intensity_out (D,)`;
- `mean_intensity_minus_source_out (D,)`;
- `total_opacity_out (D,)`;
- `scattering_fraction_out (D,)`;
- one scalar `surface_second_moment`.

Chapter 9 validates those monochromatic objects. It may display the exact
names of the later accumulator destinations in one handoff table, but it must
not derive or integrate them:

`rosseland_accumulator`, `radiation_energy_density`,
`integrated_eddington_flux`, `radiative_acceleration`,
`surface_radiation_pressure_constant`,
`temperature_correction_heating_derivative`,
`temperature_correction_mean_intensity_minus_source_integral`,
`temperature_correction_integrated_eddington_flux`, and
`temperature_correction_diagonal_lambda`.

Chapter 12 owns what those integrals mean and how they alter an atmosphere.

## 5. Movement 9A — From a known source to escaping radiation

### 5.1 Same extinction, different thermalization

Start from the opening pair of columns. Construct absorption and scattering
so their sum is identical at every depth. Use `source_and_alpha(...)` only
after deriving

\[
\chi_\nu=\kappa_{\nu,\mathrm{c}}+\kappa_{\nu,\mathrm{l}}
          +\sigma_{\nu,\mathrm{c}}+\sigma_{\nu,\mathrm{l}},
\]

\[
\alpha_\nu=
\frac{\sigma_{\nu,\mathrm{c}}+\sigma_{\nu,\mathrm{l}}}{\chi_\nu},
\qquad
S_{\nu,\mathrm{th}}=
\frac{\kappa_{\nu,\mathrm{c}}S_{\nu,\mathrm{c}}
      +\kappa_{\nu,\mathrm{l}}S_{\nu,\mathrm{l}}}
     {\kappa_{\nu,\mathrm{c}}+\kappa_{\nu,\mathrm{l}}}.
\]

Explain the zero-absorption fallback honestly: when the absorption denominator
is zero, the exact function uses `continuum_source` for `thermal_source`,
while the complete source is still weighted by `1 - scattering_fraction`.
Total extinction is floored at `FLOAT32_POSITIVE_FLOOR = 1e-38`, and
`scattering_fraction` is clipped into `[0,1]`.

The first code cell should print, for both columns, identical extinction and
different scattering fraction. Do not plot yet; the calculation exists to
make the missing variable undeniable.

### 5.2 Optical depth is wavelength specific

Recall rather than rederive Chapter 2:

\[
\tau_\nu(m)=\tau_{\nu,0}+\int_{m_0}^{m}\chi_\nu(m')\,dm',
\qquad
\tau_{\nu,0}=\chi_\nu(m_0)m_0
\]

for the exact synthesis caller. The implementation calls
`integrate_optical_depth(column_mass, extinction, surface_tau)` and uses the
parabolic interval policy already audited in Chapter 2.

Make three details explicit:

1. depth and `column_mass` run outermost to innermost;
2. the returned \(\tau_\nu\) must be non-decreasing inward for nonnegative
   extinction and increasing column mass;
3. the routine does not insert a hidden factor of one half in `surface_tau`.

Use one continuum wavelength and one active line-center wavelength computed
from preceding chapter code. Show their two optical-depth columns in one
compact table. The line center should reach the same \(\tau_\nu\) at smaller
column mass.

### 5.3 The formal solution is an attenuated sum of sources

Return to the Chapter 1 transfer equation, now with a source that changes with
depth:

\[
\mu\frac{dI_\nu}{d\tau_\nu}=I_\nu-S_\nu.
\]

For a known source and an outward ray \(\mu>0\), derive the surface formal
solution

\[
I_\nu(0,\mu)=I_\nu(\tau_b,\mu)e^{-\tau_b/\mu}
+\int_0^{\tau_b}S_\nu(t)e^{-t/\mu}\frac{dt}{\mu}.
\]

Each factor should be interpreted before code:

- \(e^{-t/\mu}\) is survival from depth \(t\);
- \(dt/\mu\) accounts for the longer slanted path;
- the deep boundary term becomes unimportant when \(\tau_b/\mu\) is large.

Build a tiny readable ray integral for a prescribed linear source. Compare it
with the constant-source result already known from Chapter 1 without
re-teaching that derivation.

### 5.4 Angular quadrature turns rays into moments

Define the monochromatic moments:

\[
J_\nu=\frac12\int_{-1}^{1}I_\nu(\mu)\,d\mu,\qquad
H_\nu=\frac12\int_{-1}^{1}\mu I_\nu(\mu)\,d\mu,
\]

\[
K_\nu=\frac12\int_{-1}^{1}\mu^2 I_\nu(\mu)\,d\mu .
\]

Then derive

\[
F_\nu=\int I_\nu\mu\,d\Omega=4\pi H_\nu .
\]

Use a small Gauss–Legendre quadrature as a *teaching construction*: direction
nodes \(\mu_q\) and weights \(w_q\) replace angular integrals by weighted
sums. Verify for an isotropic field that \(H_\nu=0\) and
\(K_\nu/J_\nu=1/3\).

Do not claim that these teaching nodes are stored inside
`transfer_tables.npz`. The production runtime uses precomputed moment
operators and surface weights; it does not expose an angle axis. The
quadrature cell explains what those operators summarize.

### 5.5 Boundary conditions belong to the solution

Place the first original schematic here, before the production operator:

- the upper boundary has no incident stellar radiation from space in the
  standard surface solution;
- radiation generated inside travels both inward and outward;
- deep layers approach the diffusion regime, where the source changes slowly
  over one mean free path.

The schematic must distinguish the physical atmosphere boundary from the
first stored layer. The exact numerical surface seed is
`extinction[:, 0] * column_mass[0]`, so the first stored layer may already
have positive optical depth.

Do not invent undocumented angular nodes or an explicit bottom-ray intensity
for the packaged operator. Explain that the fixed operators encode the
validated angular formal solution and boundary policy, while the runtime
handles layers deeper than its fixed grid through the explicit diffusion
continuation described later.

### 5.6 Contribution functions answer “where did this light come from?”

For the absorption-only teaching integral, define a ray contribution density

\[
C_I(t,\mu)=S_\nu(t)e^{-t/\mu}/\mu
\]

and an angularly integrated flux contribution constructed from the declared
quadrature. Plot one continuum row and one line-center row against
`column_mass` in a single professional panel.

The plot makes one claim: the active line center forms farther outward than
the neighboring continuum for this supplied atmosphere. State that a
contribution function is broad; its maximum is not an infinitely thin
formation surface. Mention Eddington–Barbier only as the useful interpretation
\(I_\nu(0,\mu)\approx S_\nu(\tau_\nu=\mu)\) for a smooth, nearly linear source,
not as the production algorithm.

## 6. Movement 9B — Scattering makes the source implicit

### 6.1 The source depends on the field it creates

Return to the opening columns and write

\[
S_\nu=(1-\alpha_\nu)S_{\nu,\mathrm{th}}+\alpha_\nu J_\nu.
\]

The causal loop is

```text
guess S_nu
   -> formal transfer
   -> J_nu = Lambda[S_nu]
   -> updated S_nu
   -> repeat
```

Place the second original schematic here. It should make the feedback loop
visible without implying time evolution: the iteration is a numerical fixed
point for a static radiation field.

Use a three-depth toy Lambda matrix before loading any table. Calculate one
Jacobi-style source update to establish the fixed-point equation, then derive
the actual backward Gauss–Seidel correction:

\[
\Delta S_i=
\frac{\alpha_i(\Lambda S)_i
      +(1-\alpha_i)S_{{\rm th},i}-S_i}
     {1-\alpha_i\Lambda_{ii}}.
\]

Walk the depth index from deepest to outermost. Updated deeper source values
are used immediately in subsequent shallower updates. This is why the depth
loop is sequential even though wavelength rows remain independent.

### 6.2 A common 51-point optical-depth grid makes one operator reusable

Every wavelength has a different \(\tau_\nu(m)\). The static operator expects
the shared `transfer_optical_depth_grid`, which has 51 points from 0 to 20.
Use `_interpolate_to_transfer_grid(...)` to remap `thermal_source` and
`scattering_fraction` from each physical-depth row onto that grid.

Explain rather than hide the interpolation decisions:

- `torch.searchsorted` chooses the physical-depth bracket independently for
  every target optical depth;
- the first two bracket ordinals use a linear stencil;
- the deepest edge uses a backward stencil;
- supported interior points blend forward and backward parabolas;
- curvature weights are detached because stencil choice is discrete;
- zero denominators in inactive boundary branches are replaced only to keep
  the computation finite.

Transfer-grid points above the first stored layer
(`transfer_depth_grid < optical_depth[:, 0]`) receive the first layer's
thermal source and scattering fraction. This is an explicit numerical
boundary rule, not extrapolation from nonexistent atmospheric layers.

Show one remapped source row and verify exact source-grid points plus the
above-atmosphere replacement. Do not repeat Chapter 2's derivation of
parabolic coefficients.

### 6.3 The packaged operators are static physical input

Only now load `TransferTables.from_npz(...)`. Explain each object:

| exact field | shape | meaning |
| --- | ---: | --- |
| `transfer_optical_depth_grid` | `(51,)` | common \(\tau_\nu\) coordinate |
| `mean_intensity_operator` | `(51,51)` | maps a complete source row to \(J_\nu\) |
| `mean_intensity_diagonal` | `(51,)` | derived at load time by `torch.diagonal(...).contiguous()` |
| `surface_eddington_flux_weights` | `(51,)` | maps the converged source to surface \(H_\nu\) |

The source archive stores these arrays as float64, but `from_npz(...)` moves
the consumed fields directly to the requested device and dtype. The stored
`reference_column_mass (80,)` and `__meta__` are data/provenance fields; they
are not `TransferTables` runtime fields and must not be smuggled into the
calculation.

A compact operator check should report shapes, dtype, device, a finite
diagonal, and the response to a positive source vector. It should not print a
51-by-51 matrix.

### 6.4 Eight backward sweeps are the synthesis algorithm

Call `solve_scattering_source(...)` only after the hand update matches. The
exact synthesis behavior is:

- `lambda_operator`, its diagonal, `scattering_fraction_grid`,
  `thermal_source_grid`, and the evolving source are cast to float32 on every
  backend;
- the source starts from `thermal_source_grid`;
- depth is traversed from index 50 down to 0;
- the source is floored at `1e-38`;
- the default is exactly `DEFAULT_SWEEPS = 8`;
- there is no convergence test in this fixed-sweep route.

The code cell should compare the hand-computed first update with the exact
kernel on the tiny matrix, then run the real 51-point rows. Show positivity
after every sweep through a reader helper that records checkpoints without
changing the kernel arithmetic.

Plot thermal source, the source after one sweep, and the source after eight
sweeps for one scattering-rich wavelength in one panel. The interpretation
must use actual values: scattering makes the local source depend on radiation
from other depths. Do not imply that eight is a universal mathematical
convergence theorem; it is the validated fixed numerical policy of this
implementation.

### 6.5 Surface \(H_\nu\) is a weighted moment, not yet public flux

After the eight sweeps, synthesis computes

```python
source @ surface_eddington_flux_weights
```

in `float64` on CPU/CUDA and `float32` on MPS. Keep that precision island
visible. The source iteration itself remains float32 on all three backends.

Test a scalar row against the batched row implementation. Wavelength is the
parallel batch axis; the backward depth sweep is the ordered axis. No spectrum
should move to the host here.

### 6.6 Saturated cores need an explicit second route

The fixed transfer grid ends at optical depth 20. A row is marked saturated
when its first physical layer already satisfies

```text
optical_depth[:, 0] > transfer_optical_depth_grid[-1]
```

The exact inequalities matter. Direct `_solve_flux_rows(...)` calls default
to the strict guard and raise `NotImplementedError`. The shipped synthesis
pipeline explicitly passes `assert_no_saturated_core=False`, allowing the
fallback.

Explain the fallback from the deep diffusion relation

\[
H_\nu \simeq \frac{1}{3}\frac{dS_\nu}{d\tau_\nu}.
\]

The exact `_saturated_core_flux(...)` uses the slope-limited row derivative,
updates the scattering source through a curvature correction, allows at most
`MAX_ITER = 51`, and stops a row when the summed relative change is below
`ITER_TOL = 1e-5`. It also detects unstable surface spacing when the smallest
deep step is less than \(10^{-4}|\tau_{\nu,0}|\).

Construct one ordinary and one saturated analytic row. Test the strict
failure first, then the allowed fallback. A one-panel plot should show
surface \(H_\nu\) as the first-layer optical depth crosses the grid limit,
with the threshold marked quietly. This is a numerical route-boundary plot,
not a claim that a real spectral line changes discontinuously at \(\tau=20\).

## 7. Movement 9C — Two branches make a physical normalization

### 7.1 Total and continuum are two transfer problems

Place the third original schematic before code. It should show the same
continuum absorption, continuum scattering, Planck source, and column mass
entering two rows:

```text
total:      continuum absorption + continuum scattering + line absorption
continuum:  continuum absorption + continuum scattering + zero line opacity
```

`solve_spectrum(...)` transposes all inputs, concatenates the total and
continuum wavelength rows, and calls `_solve_flux_rows(...)` once. The
continuum half zeros both line absorption and line scattering. The two halves
therefore share one operator launch pattern without becoming the same physical
problem.

The standard LTE call supplies `planck_source` as both continuum and line
thermal source and supplies zero line scattering. The more general exact
`source_and_alpha(...)` interface still keeps separate continuum/line source
and scattering arrays; do not erase that interface merely because the
standard call has equal thermal sources.

Check that the stacked result equals two independent calls under the declared
dtype policy. Then set line opacity to zero and predict before execution:

\[
H_{\nu,\rm total}=H_{\nu,\rm continuum},
\qquad
f_{\rm norm}=1.
\]

### 7.2 Normalize only after both fluxes exist

The exact internal ratio is

\[
f_{\rm norm}=
\frac{H_{\nu,\rm total}}{H_{\nu,\rm continuum}}.
\]

The ratio is also \(F_{\lambda,\rm total}/F_{\lambda,\rm continuum}\) at the
same wavelength because the \(4\pi c/\lambda^2\) factor cancels. It is not a
polynomial continuum, observed-spectrum normalization, or instrument model.

Plot `normalized_flux` across one compact line-rich window in one panel with a
quiet horizontal line at one. Build the total and continuum inputs from the
preceding chapter code or a declared integration fixture. Do not load a
golden spectrum until after the reader's calculation exists.

### 7.3 Convert \(H_\nu\) to \(F_\lambda\) without losing units

Derive the complete chain once:

\[
F_\nu=4\pi H_\nu,
\qquad
\left|\frac{d\nu}{d\lambda_{\rm nm}}\right|
=\frac{c_{\rm nm\,s^{-1}}}{\lambda_{\rm nm}^2},
\]

\[
F_\lambda\ {\rm per\ nm}
=4\pi H_\nu\frac{c_{\rm nm\,s^{-1}}}{\lambda_{\rm nm}^2}.
\]

Use the exact public helper
`_surface_flux_per_wavelength_nm(wavelength_nm,
eddington_flux_per_frequency)` for a small diagnostic array. It converts both
inputs to NumPy float64 and rejects nonpositive wavelengths. The full
spectrum's host conversion remains Chapter 10's final boundary; this compact
unit check does not authorize an early device-to-host transfer inside the
pipeline.

The visible check must independently verify:

- the factor \(4\pi\);
- the inverse-square wavelength Jacobian;
- per-nanometre units using `LIGHT_SPEED_NM_PER_S`;
- identical normalized flux before and after conversion.

Do not use the phrase “flux” alone in prose after this point. Say `H_nu`,
`F_nu`, or `F_lambda`, and state per Hz or per nm.

## 8. Movement 9D — The same physics serves a different atmosphere product

### 8.1 What the atmosphere lane shares

The fourth original schematic should compare only transfer-specific
dependencies:

```text
Torch synthesis:
    wavelength rows batched on selected device
    -> 8 fp32 source sweeps
    -> surface total/continuum H_nu

Numba atmosphere:
    independent frequency chunks on CPU
    -> ordered single-frequency depth solve
    -> J_nu, H_nu, J_nu-S_nu, K_nu at physical depths
```

Both lanes share:

- \(d\tau_\nu=\chi_\nu\,dm\);
- the thermal-plus-scattered source equation;
- a 51-point grid spanning 0 to 20;
- byte-identical `transfer_optical_depth_grid` and
  `surface_eddington_flux_weights` arrays;
- float32 operator/source-iteration islands;
- independent wavelength/frequency work outside an ordered depth recurrence.

The two stored `mean_intensity_operator` arrays are **not** byte-identical:
four float64 entries differ by at most \(10^{-8}\), and they remain different
after float32 conversion. Each lane must therefore load and validate its own
table archive rather than reconstructing one from the other. Shared physics
does not justify one fake universal API, table product, or axis order.

### 8.2 What the atmosphere lane does differently

The exact `RadiativeTransferTables` contains:

| exact field | shape | load dtype |
| --- | ---: | --- |
| `surface_eddington_flux_weights` | `(51,)` | float32 |
| `second_moment_weights` | `(51,)` | float32 |
| `transfer_optical_depth_grid` | `(51,)` | float64 |
| `mean_intensity_operator` | `(51,51)` | float32 |
| `eddington_flux_operator` | `(51,51)` | float32 |

The atmosphere source iteration runs on the fixed grid for at most 51 sweeps
and may stop early when every relative source correction is at most
`1e-5`. The remapped thermal source is initially floored at `1e-38`;
correction denominators, relative-error source scales, and updated sources use
the separate `1e-37` guard/floor policy. This is **not** the synthesis lane's
fixed eight-sweep policy.

When the physical atmosphere extends deeper than the part mapped to the fixed
grid, the atmosphere lane:

1. maps the fixed-grid solution back to the covered physical layers;
2. continues into deeper layers using
   \(H_\nu=(1/3)dS_\nu/d\tau_\nu\) and
   \(J_\nu-S_\nu=dH_\nu/d\tau_\nu\);
3. iterates the deep scattering source for at most 51 updates with an
   accumulated relative-change tolerance of `1e-5`;
4. repairs an invalid nonpositive deep source or flux by resetting the deep
   source to the Planck function before continuing.

If the first physical layer is already deeper than the grid, the mapped count
is one and the diffusion branch owns the result. This is related to, but not
byte-for-byte the same implementation as, synthesis
`_saturated_core_flux(...)`.

Use one compact controlled frequency to expose
`optical_depth_out`, `source_out`, `mean_intensity_out`,
`eddington_flux_out`, and `surface_second_moment`. Do not show the full
production kernel as a giant source block.

### 8.3 Numba parallelism remains outside the depth solve

The exact atmosphere production path is compiled with
`numba.njit(cache=True, nogil=True)`. The outer
`accumulate_transfer_range_parallel(...)` uses
`numba.prange(chunk_count)` only over independent contiguous frequency
chunks. Each chunk owns private float64 accumulators; those buffers are then
summed in fixed chunk order.

Chapter 2 already explained why this avoids races. Chapter 9 adds the
transfer-specific fact: one frequency's optical-depth integration, source
sweep, and diffusion continuation remain serial and ordered. Do not insert
`prange` into any of those depth loops.

The atmosphere line slab enters this kernel as gross float32 opacity. The
runner multiplies it by the depth/frequency float64 stimulated-emission factor
before transfer. This closes the Chapter 6 atmosphere stimulation handoff and
must be checked exactly once.

Run:

- the compiled serial frequency range;
- the parallel two-chunk route;
- the same fixed two-chunk route a second time.

Report the worst absolute and relative difference and the affected output.
Require fixed-policy repeatability. Do not promise bit identity across
different chunk counts because the final floating-point grouping changes.
Cold/warm Numba timing was taught in Chapter 2; this chapter needs only a
short warm transfer measurement if it helps interpret the actual kernel.

## 9. Visible cell ledger: 17 substantial, bite-sized cells

The target is 17 visible code cells. Small setup-only imports do not count as
substantial. Each visible cell should usually remain 10–30 lines and must
answer one question.

| cell | causal purpose | required visible result |
| ---: | --- | --- |
| 1 | build equal-extinction absorption-rich and scattering-rich columns | equal \(\chi_\nu\), unequal `scattering_fraction` |
| 2 | run `source_and_alpha(...)` and verify units/ranges | extinction floor and \([0,1]\) fraction check |
| 3 | reuse `integrate_optical_depth(...)` on continuum and line center | monotone `(2,D)` optical depth and surface seed |
| 4 | integrate one prescribed source along a few rays | variable-source formal intensity and constant-source limit |
| 5 | perform small angular quadrature | isotropic \(H=0\), \(K/J=1/3\), and \(F=4\pi H\) |
| 6 | compute and plot continuum/line contribution functions | one professional formation-depth panel |
| 7 | load and validate synthesis `TransferTables` | exact fields, shapes, dtype, device, hash |
| 8 | remap source and scattering fraction to 51 points | exact-point and boundary replacement checks |
| 9 | reproduce one Gauss–Seidel correction by hand | hand update equals kernel update |
| 10 | record one and eight float32 source sweeps | positivity plus one professional source-depth panel |
| 11 | compute surface \(H_\nu\) for scalar and batch rows | scalar/batch parity and precision-island report |
| 12 | trigger strict saturated failure and allowed fallback | route trace plus one threshold panel |
| 13 | stack total and continuum solves | stacked result equals two calls |
| 14 | remove line opacity and form `normalized_flux` | total=continuum and ratio=1 |
| 15 | solve one line-rich compact window | one professional normalized-flux panel |
| 16 | check \(H_\nu\rightarrow F_\nu\rightarrow F_\lambda\) | factor-\(4\pi\), Jacobian, unit, and ratio checks |
| 17 | run one-frequency atmosphere moments and serial/parallel mini-batch | dual-lane contract table and measured parity |

If one cell grows beyond the hard 80-line limit, move exact implementation
into the progressive package or `book/chapter09_runtime.py` and display only
the relevant bite-size function. Do not hide physical inputs or comparison
order in a notebook helper.

The chapter should have four visible movements, but remain one navigation
chapter:

1. known-source formal transfer;
2. implicit scattering source;
3. total/continuum flux semantics;
4. exact synthesis/atmosphere lane boundaries.

## 10. Original schematic and quantitative-figure contract

### 10.1 Original conceptual schematics

Create four original assets through
`scripts/textbook_schematic_specs.py`. Use the website-inspired prompt
architecture but no website image or copied composition.

1. **`ch09-rays-moments-boundaries-v1`**  
   A plane-parallel layer stack with outward and inward rays, \(\mu\), a clear
   top boundary, and three small \(J/H/K\) weighted summaries. It prepares the
   angular-moment equations and shows that the first stored layer is not
   empty space.

2. **`ch09-scattering-fixed-point-v1`**  
   A circular dependency
   `source -> formal transfer -> mean intensity -> source` with the local
   thermal term entering once. Arrows should communicate a numerical
   fixed point, not a photon orbit or time sequence.

3. **`ch09-total-continuum-stack-v1`**  
   One physical state feeding two transfer rows: total includes lines;
   continuum zeros line terms. The two rows reunite only at the ratio.

4. **`ch09-two-transfer-lanes-v1`**  
   A restrained comparison of wavelength-batched Torch surface flux and
   frequency-chunked Numba atmosphere moments, with ordered depth arrows in
   both. Avoid a full GPU architecture diagram; Chapter 10 owns caches,
   resident window state, and host movement.

Each asset needs an owned prompt, source/generation provenance, alt text,
caption, native and notebook-width review, file hash, and explicit
“conceptual schematic” caption. No generated schematic is evidence for a
numerical parity claim.

### 10.2 Professional one-panel plots

Use the shared `book.plot_style` palette and typography. Every plot is
generated from declared chapter code, has a white background, inward ticks,
physical axis units, and a paragraph interpreting actual values.

Required:

1. continuum and line-center contribution versus `column_mass`
   [g cm\(^{-2}\)]—one formation-depth claim;
2. thermal, one-sweep, and eight-sweep source versus optical depth—one
   scattering-feedback claim;
3. emergent \(H_\nu\) across the saturated-route boundary—one numerical
   route claim;
4. `normalized_flux` across one compact line-rich window—one physical
   total/continuum claim.

Do not combine these into a four-panel dashboard. A parity ledger is more
legible as a compact table than as a residual plot unless a residual curve
reveals a wavelength-dependent failure.

## 11. Self-contained source and data requirements

### 11.1 Pinned source identities audited for this contract

The implementation audit used:

| pinned file | SHA-256 | Chapter 9 ownership |
| --- | --- | --- |
| `payne_zero_synthesis/radiative_transfer.py` | `52e0d1a0c4a2713294ce1b43130c5d900e54c4cf1f8b2b05058fc2d6831ff62b` | Torch optical depth, remap, scattering, saturation, total/continuum solve |
| `payne_zero_synthesis/api.py` | `77718303c1e0052a520ece7fab277b3b1922c21d09b35a288596592d03310940` | \(H_\nu\rightarrow F_\lambda\) public conversion |
| `payne_zero_atmosphere/radiative_transfer.py` | `df8970ca629487537a7c4849278eab5d755b527002d8fc58360c9264a3aa45db` | NumPy depth helpers and atmosphere table loader |
| `payne_zero_atmosphere/transfer_kernels.py` | `50e759a085e6aefdb7819a3dbe3ef5e83405834f4b07e0a4de2f3c0e7354d3b9` | Numba monochromatic moments, deep continuation, frequency chunks |

The textbook staging may contain progressive subsets, but every displayed
exact function must be byte-identical or AST-verified against the pinned
definition it claims to teach. A simplified teaching ray quadrature must be
labelled teaching-only and may not be named like a production interface.

### 11.2 Static transfer tables

Keep the two product tables separate even though several arrays are
byte-identical:

| static file | SHA-256 | exact stored arrays |
| --- | --- | --- |
| synthesis `transfer_tables.npz` | `64f75de9af02697c0b97b7bbf919f6fed9d646622f18859eb5f66ff66e7f7a7b` | `transfer_optical_depth_grid (51,) float64`, `mean_intensity_operator (51,51) float64`, `reference_column_mass (80,) float64`, `surface_eddington_flux_weights (51,) float64`, `__meta__ (208,) uint8` |
| atmosphere `radiative_transfer_tables.npz` | `d69fcad9e22dd8dd42634e5720df717f0298849d98ee2cd93236009e22391e56` | `second_moment_weights (51,) float64`, `transfer_optical_depth_grid (51,) float64`, `mean_intensity_operator (51,51) float64`, `eddington_flux_operator (51,51) float64`, `surface_eddington_flux_weights (51,) float64` |

The transfer grids and surface-flux weights are byte-identical across the two
archives. The mean-intensity operators are not: four entries differ, with a
maximum absolute difference of \(10^{-8}\). This is an executable table
identity boundary, not a reason to average or “clean up” either array.

The atmosphere loader intentionally casts its operator/weight arrays to
float32 while retaining the transfer grid in float64. The synthesis loader
casts consumed arrays to the requested work dtype/device and derives
`mean_intensity_diagonal`; neither behavior should be inferred from the
stored dtype alone.

These are static physical/numerical inputs, not goldens. Their manifest
entries must record commit, source path, hashes, arrays, shapes, dtypes,
roles, and units/conventions.

### 11.3 Chapter inputs and comparison data

Use:

- analytic source/extinction rows generated live in code;
- one compact continuum-plus-line window computed from Chapters 5–8;
- four checksum-bound upstream atmosphere/opacity integration fixtures only
  if rerunning preceding chapters would make the notebook unreasonably slow;
- separate pinned transfer goldens loaded only after the textbook solve.

An upstream opacity slab fixture must state exactly which quantities were
already computed, their units and stimulation lifecycle, and that Chapter 9
does not claim to have built them. Do not mix upstream slabs and Payne Zero
golden transfer outputs in one `.npz`.

The normal notebook must not require `/Users/ysting/payne-zero`, the paper
tree, or the optional multi-gigabyte catalog bundle.

## 12. Main-text checks and compact parity gates

Checks appear beside the claim they test. The final notebook may summarize
them in one short ledger, but the summary must not be the first time a result
is interpreted.

### 12.1 Analytic and dimensional gates

- \(\chi_\nu\,dm\) is dimensionless.
- Identical absorption-plus-scattering sums give identical optical depth even
  when their source behavior differs.
- Optical depth is finite, nonnegative, and non-decreasing inward.
- The known-source ray integral reproduces the Chapter 1 constant-source
  limit.
- Isotropic angular quadrature gives \(H_\nu=0\) and \(K_\nu/J_\nu=1/3\).
- The formal solution remains positive for a positive source.
- Increasing line opacity moves the selected formation contribution outward.

### 12.2 Source and transfer-grid gates

- `source_and_alpha(...)` matches an independent scalar calculation.
- `scattering_fraction` lies in `[0,1]`; the zero-absorption fallback is
  exercised.
- `integrate_optical_depth(...)` matches a scalar wavelength loop.
- transfer-table file identity, shapes, dtypes, and derived diagonal match
  their contracts;
- remapping reproduces exact source points and stays finite at both
  boundaries;
- one hand-computed backward Gauss–Seidel correction matches the code;
- source rows remain positive through each of the eight synthesis sweeps;
- zero scattering leaves the complete source equal to the thermal source
  under the fixed operator route.

### 12.3 Synthesis flux gates

- one wavelength row equals the same row inside a batch;
- strict saturated-core mode raises on an exact active case;
- allowed saturated and ordinary rows both remain finite and positive;
- stacked total/continuum transfer matches two independent calls;
- zero line opacity gives equal total and continuum \(H_\nu\) and
  `normalized_flux == 1` within the declared dtype policy;
- the factor \(4\pi\), \(c_{\rm nm\,s^{-1}}/\lambda_{\rm nm}^2\), and positive
  wavelength guard are tested independently;
- total and continuum \(H_\nu\), public \(F_\lambda\), and normalized flux
  match pinned results at selected wavelengths for hot dwarf, solar dwarf,
  low-gravity giant, and cool molecule-rich fixtures.

The four-regime comparison should be a compact table: active scattering
fraction range, saturated-row count, maximum total/continuum error, and ratio
error. A near-zero scattering case counts as evidence only when its supplied
populations and opacity predict it.

### 12.4 Atmosphere-lane gates

- gross float32 line opacity is multiplied by stimulated emission exactly
  once at the transfer boundary;
- the fixed-grid source solve and deep diffusion continuation are exercised
  separately;
- `mean_intensity = source + mean_intensity_minus_source` where the exact
  branch defines those arrays;
- surface second moment, physical-depth \(J_\nu\), \(H_\nu\), and
  \(J_\nu-S_\nu\) match a pinned single-frequency checkpoint;
- serial and one-chunk compiled results are bit-equal for the controlled
  fixture;
- fixed two-chunk repetitions are identical;
- serial versus multi-chunk differences satisfy a predeclared,
  dtype- and chunk-policy-specific tolerance rather than a universal
  “machine precision” claim.

### 12.5 Device and dtype gates

Always validate CPU float64 work plus the intentional float32 source
iteration. If available, measure CUDA and MPS rather than simulate them.

- CPU/CUDA work dtype is float64; MPS work dtype is float32.
- `solve_scattering_source(...)` is float32 on every backend.
- surface-weight multiplication is float64 on CPU/CUDA and float32 on MPS.
- transfer tables remain on the selected device during synthesis.
- discrete remap brackets and detached curvature weights are checked as
  decisions, not differentiated claims.
- no full spectral array crosses to host inside Chapter 9 transfer.

Backend tolerances must be measured and declared before acceptance. An
unavailable backend receives no claim.

## 13. Redundancy and deferral audit

The chapter must not:

- rederive the Planck function from Chapter 1;
- repeat Chapter 2's parabolic-coefficient algebra, generic Numba tutorial,
  or private-buffer race-condition lesson;
- repeat any continuum, line-strength, profile, catalog, or molecular-opacity
  derivation from Chapters 5–8;
- introduce opacity through unexplained arrays or a hidden future pipeline;
- paste the long atmosphere transfer kernel into a markdown cell;
- call fixed scattering-source iteration an NLTE population solve;
- claim type-3 catalog routing implements PRD transfer;
- describe standard LTE line opacity as line scattering;
- regularize the atmosphere and synthesis transfer lanes into one invented
  interface or axis order;
- treat `reference_column_mass` as a runtime `TransferTables` field;
- call \(H_\nu\) `flux_total` before the \(4\pi\) and Jacobian conversion;
- apply an instrument operator or discuss fitting;
- teach Chapter 10 cache keys, `WindowInvariants`, memory-chunk constants,
  prewarming, or final host-copy architecture;
- teach Chapter 12 frequency-integrated correction physics;
- add end-of-chapter exercises.

Chapter 10 may invoke `solve_spectrum(...)` and the flux conversion, but it
must not rederive scattering, total/continuum stacking, or flux semantics.
Chapter 12 may consume the atmosphere transfer moment arrays and accumulators,
but it must not reteach the source equation or the 51-point solve.

## 14. Chapter summary and causal Chapter 10 handoff

End with `## 9.N Chapter summary`. It should answer the opening question in
plain language and contain no new terminology.

The summary must state:

1. absorption, scattering, and thermal emissivity determine extinction,
   scattering fraction, and a thermal source;
2. optical depth orders the physical layers by difficulty of escape at each
   wavelength;
3. the formal solution maps a known source into intensities and angular
   moments;
4. scattering makes the source depend on \(J_\nu\), so the fixed 51-point
   operator is solved by an ordered source iteration;
5. synthesis returns total and continuum \(H_\nu\) together and forms their
   ratio, with an explicit saturated-core route;
6. \(F_\nu=4\pi H_\nu\) and
   \(F_\lambda=4\pi H_\nu c_{\rm nm\,s^{-1}}/\lambda_{\rm nm}^2\);
7. the atmosphere lane uses the same physics but returns depth-dependent
   moments for later frequency-integrated structure corrections.

Then state the exact output now available:

```text
prepared depth-wavelength opacity + one structured atmosphere
    -> device-resident total H_nu
    -> device-resident continuum H_nu
    -> device-resident normalized_flux
    -> checked public F_lambda-per-nm conversion semantics
```

The unresolved problem is architectural rather than physical. A useful broad
spectrum still needs reusable catalog/window state, exact component order,
bounded memory, cache identity, backend precision policy, device-side context
cropping, and one controlled host crossing.

Close with:

### Next: keep the whole spectrum on the right device

> The transfer operator can now turn prepared opacity slabs into a physical
> native spectrum. A broad calculation must still assemble those slabs in the
> right order, reuse everything that does not depend on the star, bound its
> memory, and cross to the host only once. [Chapter 10](/reader.html?ch=10)
> composes that complete CUDA/MPS/CPU synthesis pipeline without re-deriving
> the transfer physics.
