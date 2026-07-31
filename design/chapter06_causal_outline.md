# Chapter 6 causal outline — One Spectral Line

Status: pre-implementation pedagogical and neighbor contract
Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`
Exact scientific authority:
`design/chapter06_exact_source_contract.md`

## 1. Central decision

Chapter 6 should be one chapter with three movements:

1. **One bound transition acquires an integrated strength.**
2. **Motion and finite lifetimes redistribute that strength into a profile.**
3. **One readable line becomes a checked depth-by-wavelength opacity slab.**

The causal spine is:

```text
one narrow opacity excess above a trusted smooth continuum
  -> two bound energy levels
  -> photons with the matching energy
  -> excitation-weighted population factor available to the transition
  -> gf sets integrated interaction strength
  -> stimulated emission subtracts the reverse LTE process
  -> thermal motion + microturbulence set the Gaussian width
  -> radiation + electron + neutral collisions set Lorentz wings
  -> normalized Voigt/Harris profile redistributes fixed strength
  -> one depth
  -> every depth
  -> exact CPU-atmosphere and device-synthesis boundaries
  -> unresolved problem of choosing and accumulating many heterogeneous lines
```

This order prevents five common pedagogical failures:

- a Voigt formula does not appear before the reader knows what is being
  broadened;
- `oscillator_strength` is not mistaken for bare \(f\) when the source value
  is \(gf\);
- a partition-normalized ion population is not called a bound-level
  population;
- profile normalization is not confused with a sum of truncated wavelength
  pixels;
- the atmosphere and synthesis kernels are not presented as identical merely
  because they implement the same ordinary-line physics.

There is no exercise section. Every limiting case or perturbation is performed
where it resolves a live question and is interpreted before the narrative
moves on.

The assumed reader knows algebra, elementary calculus, and the basic idea that
atoms have discrete energy levels. No prior stellar-atmosphere or
spectral-synthesis course is assumed. Every new spectroscopy term is defined
in ordinary language before its production name appears.

## 2. Opening and destination

### Opening observation

Begin with the same narrow wavelength neighborhood used to close the preceding
continuum construction, but keep the ordinate in quantities the reader has
actually built. Show the smooth **continuum extinction**
\(\kappa_\nu^{\rm abs}+\kappa_\nu^{\rm sca}\) as a quiet curve and add one
conceptual, narrow **line mass-absorption** bump. Equal units make the later
cutoff comparison useful; they do not make scattering an absorption process.
Ask the reader to compare a continuum point with the center of that excess:

> What physical interaction can add opacity in a narrow wavelength interval
> when no continuum process has an equally narrow edge there?

Do not begin with “line lists,” a Voigt function, a source module, or a
catalog. The first new object is a pair of bound energy levels.

Label the smooth curve “continuum mass extinction” and the added curve “line
mass absorption,” both in cm\(^2\) g\(^{-1}\); do not join the two labels with
a slash and do not label either as flux. The bump is a question-setting
illustration, not an observed absorption dip or a parity spectrum. Emergent
flux has not yet been computed. Own this visual as a small fourth schematic
rather than giving invented values quantitative axes.

State the controlled case beside the question: a supplied one-dimensional,
static, plane-parallel atmosphere in local thermodynamic equilibrium (LTE).
LTE means that each layer uses its own local temperature to set excitation and
the balance between absorption and stimulated emission. We do not allow
departures from that local equilibrium (NLTE effects). Follow one isolated,
ordinary Fe I transition, where “ordinary” means the standard Voigt/Harris
route rather than a special atomic profile. Delay the internal source type
number until the production boundary. There are no blends or magnetic
splitting. This chapter computes opacity only, not emergent intensity or flux.

### What the reader will build

State the destination in physical language:

- the excitation-weighted population factor available to one transition;
- the integrated strength of one transition;
- the line's Doppler width and collision/radiation damping;
- a normalized continuous profile;
- the line mass absorption at one depth;
- the corresponding `(depth, wavelength)` slab through a supplied atmosphere.

Only after those quantities are earned should the exact output be named:

| exact output | axes | unit | dtype/device |
| --- | --- | --- | --- |
| atmosphere `line_mass_absorption_coefficient` | `(80, atmosphere_wavelength)` | cm\(^{2}\) g\(^{-1}\) | NumPy `float32`, gross/pre-stimulated |
| synthesis line slab | `(6, synthesis_wavelength)` | cm\(^{2}\) g\(^{-1}\) | Torch `float32` on the selected device, gross or once-stimulated net |

These dimensions remain provisional until both artifacts pass publication.
The two slabs are validated against their own lane-specific authorities; they
are not cross-lane equality targets. The chapter does not compute emergent
flux. It does not yet know which source records belong in a window or how to
deposit overlapping lines safely.

## 3. Exact handoff into the chapter

### Supplied inputs

The chapter begins with a compact contract, not a recap lecture.

| supplied object | meaning | exact name / axes | unit | early teaching owner | what Chapter 6 may do |
| --- | --- | --- | --- | --- | --- |
| normalized ion-stage population | \(n_{s,r}/U_{s,r}\) | `partition_normalized_populations (D,6,139)` | cm\(^{-3}\) | CPU NumPy `float64` | select one line's species/stage |
| mass density | \(\rho\) | `mass_density (D,)` | g cm\(^{-3}\) | CPU NumPy `float64` | convert number interaction to mass opacity |
| fractional Doppler support | \(\Delta v_D/c\) | `fractional_doppler_widths (D,6,139)` | dimensionless | CPU NumPy `float64` | convert to \(\Delta\nu_D\) or \(\Delta\lambda_D\) |
| electron density | charged-perturber scale | `electron_density (D,)` | cm\(^{-3}\) | CPU NumPy `float64` | build Stark damping |
| neutral collision proxy | exact H/He/H2 temperature-scaled proxy | `collision_density_proxy (D,)` | cm\(^{-3}\) | CPU NumPy `float64` | build van der Waals damping |
| temperature conversion | \(hc/kT\) | `hc_over_kt (D,)` | cm | CPU NumPy `float64` | build the dimensionless excitation exponent |
| smooth extinction background | continuum absorption + scattering | `continuum_opacity (D,W)` | cm\(^2\) g\(^{-1}\) | CPU NumPy `float64` teaching view | one-line cutoff comparison only |
| LTE reverse-process factor | \(1-e^{-h\nu/kT}\) | supplied stimulated-emission factor `(D,W)` | dimensionless | CPU NumPy `float64` teaching view | apply exactly once at the lane-specific boundary |

No Saha solve, partition sum, molecular equilibrium, continuum component, edge
interpolation, or schema construction runs in the chapter. The packed
atmosphere counterparts remain opaque until the late atmosphere production
cell; the opening does not teach a second storage layout. \(D\) is depth,
\(W\) is the wavelength axis of the relevant teaching view, and depth index
zero is outermost.

### Writes handed to Chapter 7

Chapter 6 produces:

- a readable one-record ordinary-line representation;
- scalar and dense-array line-strength/profile helpers;
- manifest-bound use of the supplied Harris/FASTEX tables plus exact branch
  tests;
- one checked CPU atmosphere line slab;
- one checked Torch synthesis line slab;
- a route table stating where stimulation and float32 deposition occur.

Chapter 7 consumes these building blocks. It must not rederive the ordinary
line equation, Doppler width, damping ratio, or Harris profile.

## 4. Movement I — One transition acquires strength

Movement question:

> Before a line can have a width, what determines how much opacity it has to
> distribute?

### 6.1 A bound-bound transition selects one photon energy

Start with two discrete bound energies \(E_l<E_u\). A photon can drive the
transition when

\[
h\nu_l=E_u-E_l,\qquad
\lambda_l=\frac{c}{\nu_l}.
\]

“Bound-bound” should be defined in plain language: the electron remains
attached to the atom before and after the transition. Contrast this once with
the already-built bound-free continuum, where the final electron is unbound.

Before the first code cell, translate the source coordinate. The catalog
stores term wavenumbers
\(\tilde E_l=E_l/(hc)\) and \(\tilde E_u=E_u/(hc)\), both in cm\(^{-1}\),
rather than energies in erg. Therefore the executable wavelength bridge is

\[
\lambda_l[{\rm nm}]
=\frac{10^7}
{\left|\tilde E_u-\tilde E_l\right|[{\rm cm}^{-1}]}.
\]

The factor \(10^7\) converts cm to nm. This definition must appear before the
source values are consumed.

Use original schematic 1 before the equation:

**Two levels, one resonant photon.** A lower level, upper level, an upward
arrow labeled \(h\nu_l\), and a narrow wavelength marker. The composition is
conceptual; energy spacing and line width are not to scale.

Use the checksum-bound ordinary Fe I teaching record from raw source row
873702. Its source-row identity and static-input role remain visible, while
packing, builder hashes, and authorization stay in the data note and
verification ledger. Use only the fields needed now:

- the two energy columns;
- the zero shift/correction facts;
- the derived wavelength.

The first visible cell should reproduce
`wavelength_nm = 499.03411946178176` from the energy difference and report the
small, expected distinction from `stored_wavelength_nm = 499.0341`.

Immediate interpretation:

- the rest wavelength comes from the energy separation;
- this cell has not yet determined how deep or broad the line is;
- the source row is static input, not a computed atmosphere or comparison
  output.

Do not teach fixed-width parsing or energy-shift subfields. The chosen row has
zero shifts precisely so the first physical relation remains visible.

### 6.2 The ion stage is not the lower level

The supplied `partition_normalized_populations` tells how many particles
belong to an ion stage after division by its partition function. It does not
tell how many occupy this line's lower level.

For a lower level with statistical weight \(g_l\), the conceptual population
would be

\[
n_l=
\frac{n_{s,r}}{U_{s,r}}\,
g_l\,\exp\left(-\frac{E_l}{k_{\rm B}T}\right).
\]

Define **statistical weight** with one sentence: it counts quantum states that
share the same listed energy. The partition-normalized population is supplied;
do not derive its partition sum again.

The selected record does not supply \(g_l\) and \(f_{lu}\) separately, so the
code cannot isolate this conceptual \(n_l\). Instead introduce the source
convention:

\[
(gf)_l=g_l f_{lu}.
\]

The exact field is called `oscillator_strength`, but its ordinary catalog
value is \(gf\). Therefore the executable production factor is

\[
\frac{n_{s,r}}{U_{s,r}}\,(gf)_l
\exp\left(-\frac{E_l}{k_{\rm B}T}\right),
\]

which equals \(n_lf_{lu}\), not an independently recovered lower-level
population and not \(n_l(gf)\), which would count \(g_l\) twice.

The stored `lower_excitation_cm` is a spectroscopic wavenumber
\(\tilde E_l=E_l/(hc)\) in cm\(^{-1}\). Make the code's unit cancellation
explicit:

\[
\tilde E_l[{\rm cm}^{-1}]\,(hc/kT)[{\rm cm}]=E_l/(kT).
\]

Define **oscillator strength** here as a dimensionless measure of how strongly
the transition couples to radiation. It controls an integrated absorption
strength; it is not itself the probability that one particular photon will be
absorbed.

The second visible cell should:

1. compute \(gf=10^{-0.86}\);
2. select one solar-like depth's Fe I normalized population;
3. compute the Boltzmann exponent `lower_excitation_cm * hc_over_kt`;
4. print `excitation_weighted_partition_normalized_population_cm3` and
   `gf_weighted_excitation_factor_cm3` separately, without naming either one
   an actual lower-level population.

Interpret the values immediately. A high-excitation lower level can be rare
even when the ion stage itself is abundant.

### 6.3 Oscillator strength fixes integrated opacity

Assume LTE for one ordinary, isolated, nonmagnetic bound-bound transition.
Blends and every special-profile route remain outside this one-line equation.
Now derive the frequency-normalized LTE opacity:

\[
\kappa_{\nu,l}=
\frac{\pi e^2}{m_ec}
\frac{n_{s,r}}{\rho U_{s,r}}(gf)_l
\exp\!\left[-\tilde E_l\frac{hc}{kT}\right]\,
s_\nu(T)\,\phi_l(\nu),
\]

where

\[
s_\nu(T)=1-\exp\left(-\frac{h\nu}{k_{\rm B}T}\right).
\]

Define the constants at first use: \(e\) is the magnitude of the electron
charge in CGS electrostatic units, \(m_e\) is the electron mass in g, \(c\) is
the speed of light in cm s\(^{-1}\), \(h\) is Planck's constant in erg s, and
\(k_{\rm B}\) is Boltzmann's constant in erg K\(^{-1}\).

Define \(\phi_l\) only as “a shape whose frequency integral is one.” Its form
comes later.

Close the coordinate change before any wavelength-grid code. At wavelength
sample \(\lambda_i\), the chapter stores

\[
\kappa_{l,i}\equiv
\kappa_{\nu,l}\!\left(\nu=\frac{c}{\lambda_i}\right).
\]

This is the same mass absorption coefficient evaluated on a wavelength grid,
not an opacity density per nm. Do not multiply by
\(|{\rm d}\nu/{\rm d}\lambda|\). Profile-area checks integrate over frequency
or dimensionless \(u\), never by summing raw nm samples.

After the physical coefficient has been understood, reveal the exact synthesis
invariant:

```python
CLASSICAL_LINE_STRENGTH_COEFFICIENT = 0.026538 / 1.77245
classical_line_strength = (
    CLASSICAL_LINE_STRENGTH_COEFFICIENT * oscillator_strength / frequency_hz
)
```

The decimal expression is the parity-pinned approximation to
\((\pi e^2/m_ec)/\sqrt{\pi}\); it must not be recomputed from a new set of
rounded electron constants. The atmosphere lane uses the same literal divided
by `LIGHT_SPEED_NM_PER_S`.

Integrating over frequency reveals the causal role of \(gf\):

\[
\int\kappa_{\nu,l}\,{\rm d}\nu
\propto
\frac{n_{s,r}}{\rho U_{s,r}}(gf)_l
\exp\!\left[-\tilde E_l\frac{hc}{kT}\right]s_\nu .
\]

Strictly, \(s_\nu\) varies across the line, but an ordinary narrow-line
interpretation may evaluate its scale at the line center while the exact
production route applies it on the wavelength grid. State that distinction.

The third visible cell should double \(gf\) and the normalized
population separately, then double mass density. Print the integrated-strength
ratios 2, 2, and 1/2. This in-place dimensional sanity check belongs beside
the equation it tests. The following paragraph must read the actual ratios as
linear scaling with \(gf\) and normalized population, and inverse scaling with
mass density.

### 6.4 Reuse stimulated emission without re-teaching the continuum

Recall one established sentence: the same radiation field that drives upward
absorption also stimulates the reverse transition. Reuse \(s_\nu(T)\); do not
repeat its limiting-case derivation.

The chapter must show one exact lifecycle table:

| lane | line accumulator returns | stimulated factor applied |
| --- | --- | --- |
| atmosphere | gross, pre-stimulated line opacity | later, per frequency in the transfer accumulator |
| synthesis | net line opacity by default | at the end of `accumulate_atomic(..., apply_stim=True)` |

The next sentence should make the consequence concrete: comparing raw backend
slabs before aligning this boundary would be scientifically meaningless.

Movement I closes with one trustworthy scalar:

> We know how much opacity the transition carries, but it is still concentrated
> at one infinitely sharp frequency.

## 5. Movement II — Motion and collisions create the profile

Movement question:

> What spreads a fixed integrated strength away from the rest frequency?

### 6.5 Thermal velocities make a Gaussian core

Atoms move toward and away from the observer. Their line-of-sight velocities
Doppler-shift the rest frequency. For speeds small compared with light,
\(\Delta\nu/\nu_l\simeq v_\parallel/c\). One Cartesian component of a thermal
velocity distribution is Gaussian, so this frequency shift makes a Gaussian
line core.

Use the supplied support identity without reopening its storage layouts:

\[
\Delta v_{\rm D}^{2}
=\frac{2k_{\rm B}T}{m_l}+\xi^2,
\qquad
\delta_l=\frac{\Delta v_{\rm D}}{c}.
\]

Define **microturbulence** in the line context: \(\xi\) represents unresolved
small-scale velocities. It broadens the distribution but does not heat the
gas, move the line center, or change the absorber count.

Convert the supplied dimensionless width:

\[
\Delta\nu_{\rm D}=\nu_l\delta_l,\qquad
\Delta\lambda_{\rm D}=\lambda_l\delta_l
\]

for the narrow-line approximation.

The next visible cell should plot a one-panel set of normalized Gaussian cores
for the same Fe I line under:

- thermal broadening at two temperatures;
- one realistic microturbulent value.

The sole claim is that width changes while frequency-integrated area remains
fixed. Do not use a dashboard. The paragraph after the plot should quote the
actual measured widths and numerical area error.

This section shows the new line-profile consequence of the supplied
`fractional_doppler_widths`; it is not a second lesson on packed versus public
mass tables.

### 6.6 Finite lifetime and perturbers make Lorentzian wings

Ask why a pure Gaussian becomes too small far from line center.

Introduce the three broadening families in physical order:

1. **Radiative damping:** a finite state lifetime prevents an infinitely exact
   transition frequency.
2. **Stark damping:** charged perturbers, represented here by electron density,
   disturb the energy levels.
3. **van der Waals damping:** collisions with neutral perturbers disturb the
   levels.

Then define the exact neutral proxy once:

\[
n_{\rm pert}=
(n_{\rm H\,I}+0.42n_{\rm He\,I}+0.85n_{\rm H_2})
\left(\frac{T}{10^4\ {\rm K}}\right)^{0.3}.
\]

Do not rederive the atomic or molecular populations that enter it.

State the raw units before combining terms:
\(\gamma_{\rm rad}\) is s\(^{-1}\), while the Stark and van der Waals
coefficients are cm\(^3\) s\(^{-1}\). Independent broadening rates add because
each process contributes to the total loss of phase coherence—the loss of a
well-defined oscillation phase. The density-scaled sum is divided by
\(4\pi\Delta\nu_D\) only afterward.

The dimensionless damping ratio is

\[
a_l=
\frac{\gamma_{\rm rad}+\gamma_{\rm S}n_e+
\gamma_{\rm vdW}n_{\rm pert}}
{4\pi\Delta\nu_{\rm D}} .
\]

Immediately translate to the stored production form:

```python
damping_ratio = (
    radiative_damping
    + stark_damping * electron_density
    + van_der_waals_damping * collision_density_proxy
) / fractional_doppler_width
```

because the catalog stores each field after division by the parity-pinned
literal `12.5664 * frequency_hz`. Here `12.5664` is the implementation's
decimal approximation to \(4\pi\), not permission to replace the literal with
`4 * np.pi`. The normalized `radiative_damping` is dimensionless;
`stark_damping` and `van_der_waals_damping` have units cm\(^3\) so that
multiplication by their perturber densities is dimensionless.

The next visible cell should print the three damping-numerator contributions
at one depth, then perturb only `electron_density` and only
`collision_density_proxy`. The interpretation must identify which term moves
and which remain invariant.

Do not teach the default damping formulas. The chosen line has all three
source values.

### 6.7 The Voigt profile is a convolution, not an arbitrary blend

Use original schematic 2:

**Two causes, one profile.** A Gaussian velocity distribution and Lorentzian
lifetime/collision response feed a convolution symbol, producing a Voigt
profile with a Gaussian-looking core and Lorentzian wings. A small area ribbon
under each normalized curve shows that broadening redistributes rather than
creates strength. The curves are conceptual and not numerically scaled.

Define the convolution in words before writing it: average the Lorentzian
response over the Gaussian distribution of atomic velocities. The Voigt
profile is the result of that averaging, not a hand-chosen mixture of two
curves.

Define

\[
u=\frac{\nu-\nu_l}{\Delta\nu_{\rm D}},\qquad
\phi_l(\nu)=\frac{H(a_l,u)}{\sqrt{\pi}\Delta\nu_{\rm D}}.
\]

State both area conventions beside this definition:

\[
\int_{-\infty}^{\infty}H(a,u)\,du=\sqrt{\pi},
\qquad
\int_{-\infty}^{\infty}\phi_l(\nu)\,d\nu=1.
\]

Positive frequency offset corresponds locally to negative wavelength offset.
Use \(|u|\) when discussing production helpers that consume an absolute
offset.

Explain three limits:

- \(a=0\): Gaussian \(H=e^{-u^2}\);
- small \(a\): Gaussian core plus weak \(u^{-2}\) wings;
- large \(a\): broad Lorentzian behavior.

The next one-panel plot should show normalized profiles for three damping
ratios on a logarithmic vertical scale. The sole claim is that damping moves
opacity into the wings at fixed integrated area. The code must integrate in
frequency or dimensionless \(u\), not sum raw nm samples and call that the
profile normalization.

## 6. Movement III — One depth becomes one checked slab

Movement question:

> Can the factor-by-factor line survive translation into both working
> backends?

### 6.8 Build one transparent depth before production approximations

At one selected solar-like depth, assemble the line in this explicit order:

1. rest frequency from `wavelength_nm`;
2. \(gf\);
3. normalized population divided by mass density;
4. direct-exponential excitation;
5. fractional Doppler width;
6. three-term damping ratio;
7. normalized continuous mathematical Voigt reference;
8. stimulated-emission factor;
9. mass absorption coefficient.

The code cell should call small canonical helpers, not contain a copied
production kernel. Each intermediate should be a named scalar or one-dimensional
array with a unit comment.

Plot the resulting line opacity and the supplied continuum background on one
wavelength axis. Use a logarithmic opacity axis only if it
makes the wings legible. The sole claim is that one bound transition is narrow
and localized above a smooth background.

The paragraph after the plot must quote:

- line-center wavelength;
- Doppler width in nm and km s\(^{-1}\);
- damping ratio;
- center line-to-continuum ratio;
- the wavelength offset at which the line falls below the declared cutoff.

No emergent intensity or normalized flux is inferred.

### 6.9 FASTEX replaces one earned direct exponential

Only now ask the numerical question:

> We have evaluated \(e^{-x}\) transparently. Which table rule replaces it in
> production without silently changing a cutoff decision?

Show a compact comparison at one ordinary exponent, one half-millistep tie,
and one domain boundary. Explain the two length-1001 tables after the reader
has seen the exact exponential they approximate. The key is that the
quantization rule is part of numerical identity, not a speed claim for one
line.

Keep the exhaustive negative, nonfinite, `1001`, scalar/compiled/Torch, and
float32/float64 seam matrix in tests and the verification ledger. The visible
table needs only the representative values that change the mental model.
Explain the exact lane distinction in prose: the atmosphere table returns
zero outside its domain, while synthesis falls back to direct `exp(-x)`. The
following paragraph must identify from the actual printed values which
ordinary case is unchanged at displayed precision, which half-step is
quantized, and how the two lanes differ at the domain boundary.

### 6.10 Harris replaces the earned continuous Voigt evaluation

A general complex-error-function evaluation at every depth and wavelength is
expensive. Open the two declared table authorities just in time:

| lane | source | profile grid | dtype owner |
| --- | --- | --- | --- |
| atmosphere | 81-point source remapped to 2001 values | \(u=0\ldots10\), step 0.005 | CPU NumPy `float64` |
| synthesis | three packaged 2001-point arrays | same nominal \(u\) grid | work dtype on the selected device |

Explain the branch sequence in plain language—table core, \(u^{-2}\) tail,
intermediate blend, and asymptotic branch—then reveal
`evaluate_voigt_profile` and `interpolate_harris_profile`. Use one compact
representative table against the continuous reference. Keep all one-sided
seams and source-authority differences in strict tests; do not give a tiny
residual the visual weight of the physical profile. Interpret the measured
continuous-versus-Harris difference immediately as numerical approximation
error, not a new broadening process.

### 6.11 The synthesis shortcut and cutoff are a named arithmetic layer

The full Harris evaluator is not the final ordinary-metal deposition rule for
\(a<0.2\). Show representative center, near-wing, and far-wing values for:

- `1.0 - 1.128*a` at line center;
- `H0 + a*H1` in the near wing;
- `0.5642*a/u**2` beyond ten Doppler widths.

Add one representative cutoff/reach boundary and the float32 center-deposit
effect. Keep all 24 center reconstructions, asymmetric loop caps, and maximum
reach cases in tests. Do not present the shortcut as a new physical profile:

> Analytic normalization validates the transparent physical construction.
> Exact source parity validates the discrete production approximation.

The next paragraph must use the actual center, near-wing, far-wing, reach, and
float32 values to show where arithmetic changes while the physical causes and
line record remain fixed.

### 6.12 Extend over depth without hiding the axes

Use original schematic 3:

**One record through many layers.** A single immutable line card enters a
vertical atmosphere. Each layer supplies a different population, \(T\),
\(\rho\), \(n_e\), \(n_{\rm pert}\), and fractional width. The outputs align
as rows of one `(depth, wavelength)` slab. Depth runs outermost to innermost.

The next visible cell should broadcast the one-line calculation across the
six supplied synthesis depths. Before running it, print the contract:

```text
reads:
  line scalars
  temperature/electron_density/mass_density/collision_density_proxy (D,)
  partition_normalized_populations/fractional_doppler_widths (D,6,139)
writes:
  line_mass_absorption_coefficient (D,W)
unit:
  cm^2 g^-1
work:
  CPU NumPy float64 teaching reference
```

Use a one-panel heatmap only if the depth dependence is clearer than six
overplotted curves. The vertical coordinate should be `column_mass` or a
clearly labeled supplied depth index, with outermost/innermost direction
annotated. The color bar must name cm\(^2\) g\(^{-1}\).

Interpret why the strongest layer need not be the hottest:

- ion-stage population changes;
- the excitation weight changes with temperature while the record's lower
  excitation remains fixed;
- mass density changes the per-gram scale;
- Doppler width trades core height for width;
- collisions move opacity into wings.

### 6.13 Translate to the exact atmosphere lane

The atmosphere subsection begins with its lifecycle difference, not a source
tour.

Exact standard route:

```python
accumulate_selected_line_opacity(...) -> LineOpacityState
```

This is a production boundary, not a source tour. The provenance-bound
seven-field conversion is independently accepted; the 80-depth fixture and
oracle remain unavailable to the chapter until their separate gates close.

Only after those gates pass should the visible parity cell use the resulting
one-record `SelectedLineCatalog` and 80-depth input-only integration fixture.
Its provenance must distinguish “converted for this one-record kernel test”
from “found identically in both upstream catalogs.” It should report:

- output axes `(80,W)`;
- NumPy `float32`;
- one serial cached `njit` route because the catalog has one line;
- selected-line count;
- lane-specific pre-stimulated parity status;
- maximum difference after multiplying by the exact downstream stimulated
  array.

The fixture may contain the native 30,000-point atmosphere grid solely to
exercise this pinned route. The reader does not construct, inspect, or receive
that grid as new Chapter 6 physics; the one-depth and all-depth pedagogical
builds use the compact synthesis neighborhood above. This preserves the
accepted continuum handoff while still checking the atmosphere ordinary-line
kernel.

The visible cell should explain only one representative cutoff/reach fact and
that a one-record call uses the serial cached `njit` route. The asymmetric
101/100 loop caps, first-below-threshold rule, 8-layer gate, source-line
locations, and controlled maximum-reach cases remain mandatory strict tests
and ledger evidence, not a second implementation lesson.

Interpret the parity row immediately: it establishes the exact
pre-stimulated atmosphere route for this line and grid. It does not authorize
comparison with a once-stimulated synthesis slab, a different grid, or an
emergent spectrum.

Bulk selection, line chunks, `prange`, private buffers, and reduction order
are explicitly postponed to Chapter 7. A one-line cell must not claim parallel
speedup.

The optional detailed-transition type-0 route may appear in the strict tests
or a compact route table, but not as a second narrative implementation.

### 6.14 Translate to the exact synthesis lane

Reveal the synthesis names only after the dense result is understood:

```python
precompute_invariants(one_record_catalog, wavelength_grid_nm, runtime_device)
accumulate_atomic(
    invariants,
    state,
    do_metal=True,
    do_helium=False,
    apply_stim=True,
)
```

Here “one-record catalog” means one physical line, not a reduced mapping. The
exact constructor support entries, including empty special routes and the
three Harris arrays, are supplied by hidden manifest-verified setup and
exhaustively checked in tests. They are not a reader-facing catalog lesson.

The code cell may run only after the readable Fe I subset and synthesis
comparison artifact have passed their manifest/oracle gates. It should then
use that gate-frozen one-record mapping, the recomputed supplied continuum
background, and the same
six-depth state used by the dense helper.
It should print:

- exact population stage and element indices;
- work dtype/device;
- accumulation dtype;
- ordinary count one, auto count zero, helium absent;
- center cutoff decisions before and after FASTEX;
- output shape/device/dtype;
- lane-specific parity status.

Do not open the comparison artifact until after the reader-built dense slab
and the production call have completed.

Immediately interpret the exact precision islands:

- line strengths and damping fields enter the invariant block as float32;
- continuum cutoff, electron density, and collision proxy are float32;
- profile/population work is float64 on CPU/CUDA and float32 on MPS;
- accumulation and stimulation are float32 on every backend.

### 6.15 State what production arithmetic changed

This cell should produce one compact analytic-to-production ledger, not a
backend test suite:

| comparison | quantity reported |
| --- | --- |
| dense float64 teaching result → float32 deposit | maximum absolute and relative change on nonzero cells |
| synthesis `apply_stim=False` → `True` | exact one-factor identity |
| atmosphere pre-stim → transfer input | exact stimulated-view identity |

Keep complete loop/batched arrays, cutoff matrices, and CPU/CUDA/MPS tolerance
matrices in the verification ledger. A concise status row may say whether an
available backend was measured; unavailable hardware is “unavailable,” not
silently passed.

Read the actual ledger in the following paragraph: quote the measured
float32-deposition change, show that gross-to-net differs by the one declared
stimulated factor, and say explicitly that an unavailable backend contributes
no parity evidence.

Do not compare atmosphere and synthesis final slabs as though they share a
grid, table basis, source record, cutoff, or stimulated lifecycle. Their
scientific agreement is factor-level; their strict parity is lane-specific.

### 6.16 Four-regime evidence without four repeated lessons

Use the four supplied synthesis state regimes:

- hot dwarf;
- solar dwarf;
- low-gravity giant;
- cool molecule-rich atmosphere.

Run the same one-record synthesis route on each. Report a compact table with:

- active depth count, including the accepted `3/6`, `6/6`, `6/6`, `6/6`
  hot/solar/giant/cool result;
- peak line opacity;
- representative Doppler width;
- representative damping ratio;
- exact parity status.

The purpose is not to teach four new lines. It is to show that the same
ordinary-line construction responds honestly to different population and
perturber states. One carefully interpreted table replaces four redundant
plots.

## 7. Frozen chapter close

The reader-facing close must use this heading verbatim:

```text
## 6.17 Chapter summary
```

It should answer the opening question in seven short statements:

1. A narrow line begins with two bound levels whose energy separation fixes
   the rest wavelength.
2. The record supplies \(gf\), not \(g_l\) and \(f_{lu}\) separately, so the
   executable quantity is an excitation-weighted normalized population and
   then a \(gf\)-weighted transition factor—not an isolated \(n_l\).
3. The \(gf\)-weighted population factor and mass density set the integrated
   opacity; stimulated emission subtracts the reverse LTE process exactly once.
4. Thermal motion and microturbulence create the Doppler core.
5. Radiative lifetime, electron Stark broadening, and neutral van der Waals
   collisions create Lorentzian wings through one dimensionless damping ratio.
6. A normalized Voigt profile expresses the physics; exact Harris tables,
   FASTEX, cutoffs, and float32 deposition express the two production lanes.
7. The atmosphere route produces its own NumPy `float32` gross,
   pre-stimulated slab; the synthesis route produces Torch `float32` gross and
   once-stimulated net slabs on its own grid. Each matches its own authority;
   cross-lane slab equality is not claimed, and neither output is flux.

The next heading must also be verbatim:

```text
### Next: From one trustworthy line to an atomic forest
```

The paragraph beneath it should be causal:

> One line is now trustworthy. A real wavelength window contains thousands to
> millions of ordinary and special records, and the correct profile alone
> does not tell us which records matter or how to add them without races. The
> next chapter builds that atomic forest.

The reader link is:

`/reader.html?ch=7`

## 8. Exact notation reveal map

| concept first earned in | conceptual notation | exact name revealed afterward |
| --- | --- | --- |
| 6.1 | rest wavelength \(\lambda_l\) | `wavelength_nm` |
| 6.1–6.2 | stored lower excitation wavenumber \(\tilde E_l=E_l/(hc)\) | `lower_excitation_cm` |
| 6.2 | \(gf\) | `oscillator_strength` with its exact source convention |
| 6.2 | \(n_{s,r}/U_{s,r}\) | `partition_normalized_populations` |
| 6.2 | \(hc/kT\) | `hc_over_kt` |
| 6.2 | executable excitation/\(gf\) factors | `excitation_weighted_partition_normalized_population_cm3`, `gf_weighted_excitation_factor_cm3` |
| 6.3 | mass density \(\rho\) | `mass_density` |
| 6.4 | net LTE factor \(s_\nu\) | lane-specific `apply_stim` / downstream stimulated array |
| 6.5 | fractional Doppler width \(\delta_l\) | `fractional_doppler_widths` |
| 6.6 | electron perturber \(n_e\) | `electron_density` |
| 6.6 | neutral perturber \(n_{\rm pert}\) | `collision_density_proxy` |
| 6.6 | normalized damping inputs | `radiative_damping`, `stark_damping`, `van_der_waals_damping` |
| 6.7 | Doppler offset \(u\), damping \(a\), \(H(a,u)\) | `doppler_offset`, `damping_ratio`, continuous Voigt helper |
| 6.8 | one-depth \(\kappa_{l,i}\equiv\kappa_{\nu,l}(c/\lambda_i)\) | local direct-exponential/continuous-Voigt reference; no Jacobian |
| 6.9 | \(e^{-x}\) lookup | `FastExponentialTables`, `fast_ex`, FASTEX |
| 6.10–6.11 | production Harris and ordinary shortcut | Harris helpers and cutoff/deposit policy |
| 6.12 | all-depth slab | `line_mass_absorption_coefficient` |
| 6.13 | CPU product | `LineOpacityState`, `accumulate_selected_line_opacity` |
| 6.14 | device product | `AtomicInvariants`, `precompute_invariants`, `accumulate_atomic` |

Internal helper names should appear in code captions or exact route tables
after the physics is earned. Main prose says “the excitation exponential”
before `fast_ex`, and “the Harris approximation” before
`interpolate_harris_profile`.

## 9. Visible-code plan

Every visible cell has one conceptual purpose, normally 10–30 lines. Long
source kernels live in the progressive package and are called rather than
pasted into Markdown.

| checkpoint | live question | visible result |
| ---: | --- | --- |
| 1 | Which photon matches the two chosen bound levels? | source-row provenance and derived wavelength |
| 2 | Which excitation-weighted population factor is available, and where is \(g_l\)? | normalized population, dimensionless exponent, direct exponential, two honest runtime factors |
| 3 | What fixes the integrated area? | \(gf\), population, density, and stimulation scaling ratios |
| 4 | How does supplied motion redistribute that area? | one width perturbation, Gaussian area check, one Doppler plot |
| 5 | Which perturber owns which damping term? | three-term ledger and two isolated perturbations |
| 6 | Why is the profile a Voigt convolution? | three limits, \(H\)/\(\phi_\nu\) normalization, one damping plot |
| 7 | What does the exact transparent line look like at one depth? | direct exponential/continuous mathematical Voigt reference, factor ledger, one line-over-continuum plot |
| 8 | How does FASTEX replace the direct exponential? | one ordinary value, one half-step, one domain comparison |
| 9 | How does Harris replace the continuous profile? | compact representative approximation table |
| 10 | What shortcut and cutoff does synthesis deposition add? | center/near/far values, one reach boundary, float32 center effect |
| 11 | How does one immutable record change through depth? | `(D,W)` dense slab and one depth heatmap |
| 12 | Does the exact atmosphere route preserve the intended one-line result? | serial-`njit` output/stimulation contract and one parity row |
| 13 | Does the exact synthesis route preserve it on device? | indices/counts, output contract, and one parity row |
| 14 | What changed when transparent physics became production arithmetic? | compact float32/stimulation approximation ledger |
| 15 | Does the same construction behave honestly across stellar regimes? | four-regime compact evidence table |

Fifteen cells are frozen; sixteen is the hard ceiling. Do not compress
FASTEX, Harris, and the synthesis shortcut into one internals cell, and do not
collapse the two production lanes into false cross-lane parity.

Hidden setup may:

- load manifest-verified static data and the gate-frozen one-line subset;
- load the supplied integration fixture;
- call shared plotting style helpers;
- create dataclasses used only to present results.

It may not precompute line physics, load comparison goldens early, or import
the upstream Payne Zero checkout.

## 10. Original schematic plan

All schematics use the website-inspired white/slate/navy/beige
scientist-sketched aesthetic but have new Chapter 6 compositions and owned
prompts. Implementation should reuse the generation/provenance pattern of the
read-only website `scripts/generate_physics_images.py` through this textbook's
`scripts/textbook_schematic_specs.py`; it must not copy an existing website
image. Every final asset records its Chapter 6 prompt/specification, generator,
date, source hash, output hash, caption guard, alt text, and visual/scientific
review.

### Schematic 0 — Smooth extinction, narrow missing opacity

**Claim.** A smooth continuum cannot by itself explain a narrow opacity excess.

Composition:

- a quiet, broad continuum-extinction ribbon;
- one localized line-absorption bump above it;
- two simple wavelength markers, one off-line and one at line center;
- a question arrow leading from the bump to two bound energy levels.

Caption guard: conceptual quantities with no numerical axes; extinction and
absorption share units here but are not the same physical process.

### Schematic 1 — Two levels, one resonant photon

**Claim.** A line center comes from one energy separation, and absorption
requires an excitation-weighted population factor associated with the lower
state.

Composition:

- lower and upper horizontal energy levels;
- a small cluster of particles on the lower level;
- one upward photon arrow labeled \(h\nu_l\);
- a narrow wavelength marker connected to \(\lambda_l=c/\nu_l\);
- a quiet side label “bound before and after.”

Caption guard: conceptual energy spacing and particle counts, not to scale.

### Schematic 2 — Gaussian core plus Lorentzian wings

**Claim.** Velocity and damping are independent causes combined by a
convolution.

Composition:

- left branch: thermal + microturbulent velocity bell;
- right branch: radiative + electron + neutral damping response;
- center convolution symbol;
- output curve with labeled “core” and “wings”;
- a small equal-area ribbon below the normalized inputs/output.

Caption guard: equal area refers to the ideal continuous normalized profiles,
not a cutoff production deposit.

### Schematic 3 — One immutable record through many layers

**Claim.** The line data stay fixed while the atmosphere changes its amplitude,
width, and damping with depth.

Composition:

- one line card with \(\lambda_l,\tilde E_l,gf,\gamma\) at left;
- six or eight horizontal atmosphere layers in the center, outer to inner;
- each layer receives state tokens \(T,\rho,n_e,n_{\rm pert},n/U,\delta\);
- a narrow row profile exits each layer;
- rows align into a `(depth, wavelength)` slab at right.

Caption guard: row colors encode relative structure conceptually, not a
quantitative parity result.

No catalog-routing or sparse-deposition schematic appears here; those are
Chapter 7 originals.

## 11. Professional one-panel plot plan

| plot | sole claim | required presentation |
| --- | --- | --- |
| Doppler cores | higher thermal speed or microturbulence broadens the core at fixed integrated strength | wavelength offset in pm or velocity in km s\(^{-1}\); normalized frequency-profile ordinate |
| damping redistribution | increasing \(a\) moves opacity from core into \(u^{-2}\) wings | dimensionless \(u\); logarithmic vertical scale; numerical areas quoted |
| one-depth line over continuum | a bound transition creates localized opacity above a smooth background | wavelength in nm; cm\(^{2}\) g\(^{-1}\); line center and cutoff annotated |
| all-depth slab | one line changes with atmospheric state while its rest wavelength stays fixed | one heatmap; wavelength offset and column mass/depth, physical color-bar unit |

Use a compact Harris table rather than giving a small implementation residual
the visual weight of the physical line. Do not create a decorative fifth
figure.

Every plot:

- uses the shared professional paper-inspired style and named color palette;
- has one panel and one physical claim;
- uses a white background, inward ticks, consistent serif math typography,
  restrained guides, and no default color cycle;
- labels units and states whether the ordinate is \(H\), \(\phi_\nu\),
  continuous \(\kappa_\nu\), or sampled \(\kappa_{l,i}\);
- is interpreted from its actual numerical values in the following paragraph;
- is inspected for clipping, legend density, and notebook/mobile readability.

## 12. Chapter 5 → 6 → 7 neighbor and redundancy audit

### 12.1 The three questions form one dependency chain

| chapter | question answered | artifact handed forward |
| --- | --- | --- |
| 5 — Continuous Opacity and Scattering | What creates the smooth absorption/scattering background? | continuum absorption, scattering, derived extinction, stimulated factor, atmosphere selection threshold |
| 6 — One Spectral Line | How does one bound transition get a strength, width, damping, normalized shape, and all-depth opacity? | exact ordinary one-line building blocks and checked slabs |
| 7 — Atomic Line Forests and Special Profiles | Which records enter a window and how are ordinary/special lines routed and accumulated at scale? | complete atomic line opacity |

### 12.2 Chapter 6 may consume from Chapter 5

- `continuum_absorption + continuum_scattering` as a supplied synthesis
  background;
- the atmosphere `(D,344)` selection threshold as a supplied exact product;
- the stimulated-emission factor as already derived physics;
- the Chapter 5 state/continuum code needed to recompute the background before
  parity.

Once gate-frozen, the atmosphere parity integration fixture may carry its
native grid and packed support as opaque input-only test state. That is not an
additional Chapter 5 teaching handoff and is not used by the causal one-line
derivation.

It must not repeat:

- \(n\sigma/\rho\);
- H-minus, hydrogen/helium/metal/molecular continuum components;
- Thomson or Rayleigh scattering;
- atmosphere 30,000-point grid construction;
- synthesis edge triplets or log-parabolic interpolation;
- Chapter 5's stimulated-factor limiting-case plot.

One sentence and one lifecycle table are the complete backward-reference
budget.

### 12.3 Chapter 6 may consume from Chapter 3–4

- actual and partition-normalized population meanings;
- `fractional_doppler_widths`;
- `electron_density`, `mass_density`, and `collision_density_proxy`;
- molecular H2 only through the already-computed proxy.

It must not repeat:

- partition-function construction or Saha balance;
- packed/public layout derivations;
- molecular mass action or Newton continuation;
- the Chapter 3 thermal/microturbulent limiting-case storage test.

Chapter 6 owns how the stored fractional width changes a line profile, which
is new.

### 12.4 Chapter 7 must reuse rather than rederive

Chapter 7 may invoke:

- \(gf\)-weighted lower-level strength;
- FASTEX;
- fractional Doppler width;
- three-term damping ratio;
- the exact ordinary Harris profile and production shortcut;
- one-line cutoff/deposit primitives.

It may not rederive them. Chapter 7's first genuinely new objects are:

- full catalog records and corrections;
- line-size/window margins;
- exact selection inequalities;
- conservative atmosphere-dependent keep decisions;
- many-line center/wing accumulation;
- chunking, scatter-add, private buffers, and reduction order;
- hydrogen, helium, autoionizing, merged-series, COR, and PRD route behavior.

### 12.5 Chapter 8 remains separate

No molecular line record enters Chapter 6. Molecular populations influence the
neutral collision proxy and nothing more. Text-band, TiO, H2O, molecular
masses, source compilation, streaming, and molecular deposits stay in Chapter
8.

### 12.6 Forward/backward reference budget

Allowed references:

- opening: one phrase, “the smooth background built in Chapter 5”;
- population section: one phrase, “the normalized ion population supplied by
  Chapter 3/4”;
- microturbulence section: one sentence recalling the stored fractional width;
- exact atmosphere section: one sentence that the threshold construction is
  Chapter 5's output;
- closing: one direct need for Chapter 7.

Avoid “as we saw repeatedly” and “as we will later see.” The reader should not
need to page backward to understand the active equation.

### 12.7 Redundancy locks

| idea | sole teaching owner | Chapter 6 treatment |
| --- | --- | --- |
| photon energy and Planck radiation | Chapter 1 | one resonant energy-difference use |
| partition functions and ion populations | Chapter 3 | supplied normalized population |
| microturbulent fractional width storage | Chapter 3 | consequence for one line profile |
| molecular H2 abundance | Chapter 4 | hidden inside supplied collision proxy |
| cross section to mass opacity | Chapter 5 | invoke the established per-gram meaning |
| continuum stimulated-emission derivation | Chapter 5 | reuse once; distinguish lifecycle |
| ordinary line strength/profile | Chapter 6 | derive and implement fully |
| catalogs/selection/sparse forests/special atomic profiles | Chapter 7 | explicit deferral |
| molecular bands/source compilation | Chapter 8 | explicit deferral |
| transfer and emergent flux | Chapter 9 | absent except “not yet a spectrum” |

## 13. Flow and acceptance audit

### Causal flow

- Does the chapter open with one narrow feature the continuum cannot explain?
- Do two bound levels appear before oscillator strength?
- Is conceptual \(n_l\) introduced before line opacity, while the executable
  path is honestly limited to the available excitation-weighted and
  \(gf\)-weighted factors?
- Does integrated strength appear before width?
- Does each broadening cause enter before the Voigt convolution?
- Does the dense one-depth line appear before the all-depth and production
  slabs?
- Does every code output receive immediate interpretation?

### Accessibility

- Can a reader with introductory physics explain bound-bound, statistical
  weight, oscillator strength, microturbulence, natural damping, Stark
  damping, van der Waals damping, and convolution in plain language?
- Is every symbol defined at first use?
- Are frequency and wavelength distinctions explicit?
- Is continuous frequency normalization distinguished from a discrete
  wavelength-grid sum?
- Are cm\(^{-1}\), nm, Hz, cm s\(^{-1}\), cm\(^{-3}\), and cm\(^2\) g\(^{-1}\)
  attached before code consumes values?

### Exactness

- Is `oscillator_strength` correctly identified as \(gf\)?
- Is the lower statistical weight not counted twice?
- Is the literal classical-strength coefficient retained?
- Is `12.5664`, rather than a newly evaluated \(4\pi\), retained in damping
  normalization?
- Does the one-record synthesis mapping include empty helium metadata and all
  three manifest-verified Harris support arrays?
- Are atmosphere and synthesis FASTEX boundary rules separate?
- Are the two Harris table authorities separate?
- Is the synthesis low-damping shortcut shown?
- Are atmosphere and synthesis stimulation lifecycles separate?
- Are axes, dtype, device, cutoff, and output units stated?
- Is float32 deposition measured?
- Are strict comparisons lane-specific?

### Nonredundancy

- Is Saha/partition/molecular equilibrium absent?
- Are continuum components and interpolation absent?
- Is the Chapter 3 Doppler storage lesson not repeated?
- Is there exactly one physical line record?
- Are catalog parsing, selection, forests, and special profiles absent except
  for explicit boundary tables?
- Are molecular line formats absent?
- Is radiative transfer absent?

### Visual and pacing

- Does each schematic solve a genuine spatial or dependency problem?
- Does each plot make one claim and use professional styling?
- Are conceptual and quantitative figures labeled honestly?
- Are there no large source blocks in Markdown?
- Are there no detached exercises?
- Does the chapter remain at fifteen target / sixteen hard-ceiling
  visible cells?

### Closing

- Does the summary answer why the narrow line exists and what controls its
  strength, core, and wings?
- Does it name the exact slab without introducing a new API?
- Does the final paragraph make catalog selection and many-line accumulation
  the unavoidable next problem?
- Does it link directly to `/reader.html?ch=7`?
