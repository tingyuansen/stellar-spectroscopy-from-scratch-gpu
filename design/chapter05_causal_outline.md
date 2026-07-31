# Chapter 5 causal outline — Continuous Opacity and Scattering

Status: central authoring outline, not reader-facing prose  
Scope: complete atmosphere and synthesis continuum physics in one chapter  
Primary evidence:
`design/chapter05_atmosphere_source_audit.md`,
`design/chapter05_synthesis_source_audit.md`, and the global chapter contracts

## 1. Central decision

Chapter 5 should be one chapter with four movements:

1. **A microscopic interaction becomes mass opacity.**
2. **Named processes become an absorption and scattering budget.**
3. **Two exact consumers place that budget on two different grids.**
4. **The reader-built pieces become checked atmosphere and synthesis outputs.**

The causal spine is:

```text
one photon and one particle
  -> microscopic cross section
  -> absorber population
  -> absorption or redirection per unit mass
  -> named component budget at one depth
  -> depth-by-frequency component columns
  -> direct atmosphere sampling or synthesis edge sampling
  -> final continuum slabs
  -> the unresolved narrow-line forest
```

This order prevents three common failures.

- A list of routines never replaces the physical question each routine
  answers.
- The atmosphere and synthesis implementations are not blended into a
  fictitious common algorithm.
- Exact field and routine names enter only after the reader knows what the
  corresponding quantity means.

The chapter is self-contained at the level expected of a final-year
undergraduate. It may use basic mechanics, thermodynamics, exponentials, and
elementary interpolation. It defines every spectroscopy-specific idea when it
first appears: cross section, bound-free absorption, free-free absorption,
stimulated emission, partition-normalized population, Gaunt factor, Rayleigh
scattering, and an opacity edge.

There is no exercise section. Every useful perturbation, limit, or numerical
experiment is performed inline because it resolves the question currently
being discussed.

## 2. Opening and destination

### Opening observation

Return briefly to the already-built Chapter 1 teaching spectrum and point to
one wavelength between narrow features. Ask what can still interact with that
photon. This uses an established pedagogical object; it does not preload a
Chapter 5 calculation or open a comparison golden.

The opening question is:

> Between the narrow dips, why is the gas not transparent?

Do not begin with a table bundle, a source path, or a catalog of process names.
First let the residual structure demand a physical explanation.

### What the reader will build

State the destination in physical language before exposing implementation
names:

- an absorption coefficient per unit mass;
- a scattering coefficient per unit mass;
- an absorption-weighted thermal source for the atmosphere calculation;
- a direct atmosphere wavelength sampling;
- an edge-aware synthesis sampling that reconstructs the continuum on an
  arbitrary requested wavelength grid.

Only after Movement I earns the units and axes should the exact output names be
introduced:

| exact output | axes | unit | owner |
| --- | --- | --- | --- |
| `continuum_absorption` | `(depth, frequency)` or `(depth, wavelength)` | cm\(^{2}\) g\(^{-1}\) | atmosphere and synthesis |
| `continuum_scattering` | same | cm\(^{2}\) g\(^{-1}\) | atmosphere and synthesis |
| `continuum_source` | `(depth, frequency)` | erg s\(^{-1}\) cm\(^{-2}\) sr\(^{-1}\) Hz\(^{-1}\) | atmosphere composite and sampled synthesis diagnostic |

The standard synthesis entry returns the first two arrays, not the third.
The atmosphere standard route returns all three. That distinction is revealed
near the end of Movement I, after absorption, scattering, and a source have
separate physical meanings.

For later line selection, synthesis also forms the derived total

```python
continuum_opacity = continuum_absorption + continuum_scattering
```

This is a downstream extinction scale, not a third independently computed
physical slab.

### Honest boundary

The chapter receives two closed Chapter 4 handoffs: the packed
`ContinuumAtmosphereState` adapter for the atmosphere lane and the separate
27-field schema-v4 mapping for synthesis. The standard synthesis pipeline
projects its exact 18-field continuum view from that 27-field mapping. This
trimmed synthesis view is not the packed 18-field atmosphere adapter, and
neither 18-field view is reconstructed from the other.

The chapter constructs continuum opacity, not a spectrum. It does not yet
introduce oscillator strengths, line profiles, or a transfer iteration.
The final continuum slabs explain broad escape windows and provide a local
background scale, but they cannot create the narrow forest in the opening
spectrum.

## 3. Movement I — A microscopic interaction becomes mass opacity

Movement question:

> How does the probability for one photon to interact with one particle become
> a coefficient that describes one gram of stellar gas?

### 5.1 A cross section is an effective interaction area

Introduce a microscopic cross section \(\sigma_\nu\) as an effective area for
an interaction at frequency \(\nu\). If a volume contains \(n\) absorbers per
cm\(^{3}\), a path of length \(ds\) has interaction probability
\(n\sigma_\nu ds\) in the optically thin limit.

This earns the number absorption coefficient,

\[
\alpha_\nu=n\,\sigma_\nu
\qquad [{\rm cm}^{-1}],
\]

and then the mass absorption coefficient,

\[
\kappa_\nu=\frac{\alpha_\nu}{\rho}
          =\frac{n\,\sigma_\nu}{\rho}
\qquad [{\rm cm}^{2}\,{\rm g}^{-1}].
\]

Define every factor and perform the unit cancellation in the main text. An
inline code cell doubles \(n\), then doubles \(\rho\) while holding the other
quantities fixed. The printed factor-of-two and factor-of-one-half results are
the first physical gates, not exercises.

At this point introduce the exact state names `mass_density` and the relevant
named population field for the first example. Do not introduce the full
structured state.

### 5.2 Bound-free and free-free absorption

Use an original threshold schematic to distinguish two processes.

- In **bound-free absorption**, a photon supplies at least the binding energy
  needed to liberate an electron. The opacity is zero below the threshold and
  generally structured above it.
- In **free-free absorption**, an already free electron absorbs energy while
  accelerated in the electric field of an ion. There is no discrete
  ionization threshold, but the ion charge, electron density, temperature,
  frequency, and a quantum correction all matter.

Define the dimensionless **Gaunt factor** as the quantum correction to the
classical free-free result. It is a table or fitted function, not a new
population.

This section should use one schematic and a few equations, not a taxonomy of
all later processes. One bound-free threshold and one smooth free-free curve
are enough to establish the vocabulary.

### 5.3 Net absorption and stimulated emission

Absorption removes photons from a beam, but an excited system can also be
stimulated by the same radiation field to emit into it. Under LTE, the net
absorptive factor recurring in the continuum is

\[
s_\nu(T)=1-\exp\!\left(-\frac{h\nu}{kT}\right).
\]

Derive the factor once from upward absorption minus stimulated downward
emission. Check:

\[
s_\nu\rightarrow\frac{h\nu}{kT}\quad(h\nu\ll kT),
\qquad
s_\nu\rightarrow1\quad(h\nu\gg kT).
\]

A one-panel curve against \(h\nu/kT\) should show both limits and the allowed
range \(0\le s_\nu\le1\). The code should use a numerically stable exponential
form.

Immediately state the implementation rule: some fitted continuum
coefficients already contain an algebraically equivalent convention. The
factor is derived once, but the exact routine decides whether it must be
applied explicitly. The chapter must never multiply every component by a
second copy.

### 5.4 Absorption creates heat; scattering redirects

Before any component budget, separate three quantities:

\[
\kappa_\nu^{\rm abs},\qquad
\kappa_\nu^{\rm sca},\qquad
N_\nu=\sum_j \kappa_{\nu,j}^{\rm abs}S_{\nu,j}.
\]

\(N_\nu\) is the thermal source numerator. Scattering contributes to
extinction, but it does not enter this LTE thermal numerator as though it had
created a new photon locally.

For the atmosphere lane,

\[
S_\nu^{\rm cont}=
\frac{N_\nu}{\kappa_\nu^{\rm abs}}
\]

where absorption is nonzero; otherwise the exact composite uses \(B_\nu\).
Most active absorptive terms have \(S_{\nu,j}=B_\nu\). H I and H-minus can use
departure-coefficient corrections. The synthesis sampled diagnostic returns
\(B_\nu\); the standard synthesis continuum returns only absorption and
scattering.

Use an original schematic in which an incoming photon has two honest fates:
energy transfer to matter or redirection. The graphic must not imply that
scattering is zero extinction.

### 5.5 H-minus: the first complete continuum absorber

Explain why a weakly bound extra electron on neutral hydrogen can matter:
hydrogen is abundant, free electrons are available in warm photospheres, and
the H-minus bound-free threshold lies in a spectroscopically important range.

H-minus is not a stored population in either Chapter 4 handoff. At each depth,
the continuum route forms its local factor from temperature, electron
density, and the normalized neutral-hydrogen base
`hydrogen_partition_normalized_ion_stage_populations[:, 0]`. The atmosphere
route can also carry multiplicative hydrogen and H-minus departure
coefficients; a coefficient of one is the LTE reference case. This local
consumer calculation is not a second ionization-equilibrium solve.

Run the manifest preflight immediately before this first table-driven result.
Then build the two pieces separately.

1. H-minus bound-free uses the locally inferred population factor, an
   85-point wavelength cross-section table, its threshold
   \(1.82365\times10^{14}\ {\rm Hz}\), and the appropriate net-absorption
   convention.
2. H-minus free-free uses neutral hydrogen and electron densities with a
   \(22\times11\) wavelength/temperature table.

The raw field
`hminus_boundfree_cross_section_cm2` stores numbers in units of
\(10^{-18}\ {\rm cm}^{2}\), despite the field spelling; the physical
cross section is obtained only after the explicit factor `1e-18`. Show this
once beside the table loader.

Bound-free and free-free retain their own stored-coefficient conventions. The
chapter must not indiscriminately multiply both completed terms by an extra
copy of the generic stimulated-emission factor.

The visible calculation probes immediately below, at, and above the bound-free
edge, then plots bound-free, free-free, and their ordered sum in one panel.
The paragraph after the plot should say what changes physically and what is
merely table interpolation.

### 5.6 Hydrogen and helium build a regime-dependent background

Start with one neutral-hydrogen bound level. A partition-normalized
population is now needed, so define it locally:

\[
\widetilde n_{i,r}=\frac{n_{i,r}}{U_{i,r}(T)}.
\]

It is a convenient base population from which explicit statistical weights
and Boltzmann factors recover bound-level populations. It is not the actual
ion number density. Free-free charge partners use actual densities.

Grow the construction in this order:

1. H I explicit bound-free levels;
2. the high-level hydrogenic tail;
3. H II free-free with the charge-1 Gaunt factor;
4. H2-plus absorption;
5. He-minus absorption;
6. He I bound-free, tail, and free-free;
7. He II bound-free/tail and He III free-free.

H2-plus is not H2 collision-induced absorption. It is an ionic absorption
process built from hydrogen-stage populations. He-minus is a fitted
free-free-like process involving neutral helium and electrons.

The visible code calls short, exact component helpers for one solar-like and
one hot layer. A single panel uses restrained colour families and line styles
to show that H-minus weakens while ion/electron and helium terms become more
important in the hot state. The plot should not contain every later metal
curve.

Movement I closes with:

- a checked H/He/light-particle absorption budget on sampled frequencies;
- explicit units and axes;
- actual and partition-normalized population roles kept distinct;
- absorption, scattering, and source given separate meanings.

It does not yet claim a complete continuum.

## 4. Movement II — Complete the physical budget without hiding its owners

Movement question:

> Which additional particles shape the broad background, and which
> interactions absorb energy rather than merely changing a photon's direction?

### 5.7 Metals matter even when no metal line is centred nearby

Explain the physical family once: neutral and ionized metals possess
photoionization thresholds, and ionized metals also provide Coulomb partners
for free-free absorption. “Metal continuum” is not one constant.

Introduce components in causal groups rather than source order.

#### Neutral metals

Teach C I, Mg I, Al I, Si I, and Fe I as bound-free families. One
partition-normalized population is multiplied by explicit level weights,
Boltzmann factors, and either analytic or tabulated cross sections. A selected
near-UV residual from the H/He budget motivates these terms.

The exact consumers do not use identical neutral-metal approximations:

- the atmosphere route evaluates full C I, Mg I, Al I, Si I, and Fe I
  families;
- the standard synthesis route uses compact C I, Mg I, Al I, and Si I
  branches;
- full neutral-metal helpers belong to the sampled diagnostic/implemented
  extension, not to the standard edge-triplet composite.

State this boundary after the shared physical idea is understood. Do not
write one blended “metal routine.”

#### Lukewarm and hot ions

The atmosphere's lukewarm group contains N I, O I, C II, Mg II, Si II, and
Ca II. The synthesis precomputed sampled extension contains N I, O I, Mg II,
and Ca II; these explicit light-element grids are not standard-pipeline
terms.

The hot group has two population roles:

\[
\text{bound-free: }\widetilde n_{i,r},
\qquad
\text{free-free: }\sum_r q_r^2 n_{i,r}.
\]

The synthesis helper materializes 21 normalized hot-metal population columns
and five charge-square columns. The visible perturbation changes one
population view at a time:

- changing only the normalized view changes bound-free opacity;
- changing only the actual ion-stage view changes the charge-square
  free-free term.

This one experiment teaches the ownership rule more clearly than a long field
table.

Si II receives one short explanation as a temperature- and frequency-bracketed
table. The atmosphere generates its own cached table; synthesis consumes the
packaged Peach table. Their physical purpose overlaps, but their numerical
objects are not interchangeable.

The hot-metal 60-record bound-free table and its running one-percent acceptance
condition should be described at the exact helper boundary. Preserve the
ordered accumulation; do not reassociate it as a mathematically “cleaner”
sum.

### 5.8 Molecular continua are interactions, not a second chemistry lesson

Define the three atmosphere-only absorptive mechanisms.

- **CH photodissociation:** a continuum photon breaks CH.
- **OH photodissociation:** a continuum photon breaks OH.
- **H2 collision-induced absorption:** a collision temporarily gives an H2-H2
  or H2-He pair an absorptive dipole.

Do not rederive molecular equilibrium. The chapter needs only the supplied CH
and OH populations and the exact local H2 consumer policy.

The atmosphere adapter's `ch_population` and `oh_population` names are exact
but potentially misleading: they are aliases for packed
\(N_{\rm CH}/U_{\rm CH}\) and \(N_{\rm OH}/U_{\rm OH}\) in slots 845 and 847.
Their cross-section helpers restore the corresponding partition factors.

The atmosphere continuum deliberately reconstructs an actual H2 density for
CIA and Rayleigh:

```text
actual neutral H
  -> six-level H I partition
  -> normalized H I ground population
  -> ground departure factor
  -> square
  -> atmosphere H2 equilibrium table
  -> continuum-local H2 density
```

It clips the equilibrium-table interpolation at 19900 K and retains the
consumer value at exactly 20000 K, setting it to zero only above 20000 K.
CH and OH vanish per layer at 9000 K. A mixed cool/warm column can still enter
the molecular composite globally and evaluate CIA in warmer layers up to its
own cutoff. The H2-CIA table interpolation retains the pinned two-column
weighting order, including its counterintuitive exact-node behaviour. The
visible probe must evaluate one lower grid node, one midpoint, and the adjacent
upper node rather than silently replacing that convention with an ordinary
linear interpolation.

Show one compact feature matrix:

| process | atmosphere composite | standard synthesis composite |
| --- | --- | --- |
| CH continuum | yes | no |
| OH continuum | yes | no |
| H2-H2 / H2-He CIA | yes | no |
| H2-plus absorption | yes | yes |
| H2 Rayleigh | yes | yes, with a synthesis-specific H2 reconstruction |

The schema's stored `molecular_hydrogen_population` controls none of the
standard synthesis continuum terms. Perturbing it leaves that continuum
unchanged; perturbing neutral H changes the reconstructed H2 Rayleigh term.
Atmosphere H2-plus also uses an H I ground departure coefficient that the
synthesis H2-plus state does not carry.

This is a model boundary, not a defect the textbook should “repair.”

Disabling molecule production normally zeros the packed CH and OH populations,
but it does not switch off the continuum-local H2 reconstruction. The visible
code should compare molecule-enabled and molecule-disabled states; probe
100/101 K, 8999/9000 K, 19899/19900 K, and exactly/just above 20000 K; cross
the CIA boundary at 20000 cm\(^{-1}\); and compare an all-warm column with a
mixed cool/warm column. Plot only the cool-state CH, OH, and CIA contributions
in one infrared-focused panel.

### 5.9 Scattering is redirection with a colour dependence

Begin with Thomson scattering by free electrons:

\[
\sigma_{\rm T}=0.6653\times10^{-24}\ {\rm cm}^{2}.
\]

At fixed \(n_e/\rho\), its mass scattering coefficient is independent of
wavelength. Then introduce Rayleigh scattering as induced-dipole scattering
by neutral H, neutral He, and H2. In its ordinary domain it increases strongly
toward short wavelength.

The atmosphere and synthesis H I Rayleigh laws differ:

- the atmosphere uses a literal inverse-wavelength fit;
- synthesis uses the Gavrila/polarizability tables.

Their H2 reconstructions also differ. Match each lane to its own exact law.
Do not demand cross-lane equality.

The atmosphere frequency caps must appear in the executable boundary check:
\(2.463\times10^{15}\) Hz for H I,
\(5.15\times10^{15}\) Hz for He I, and
\(2.922\times10^{15}\) Hz for H2. Each curve rises blueward below its cap and
flattens above it because the input frequency itself is capped.

One code cell verifies Thomson constancy and the blueward Rayleigh trend. One
panel plots electron, H I, He I, and H2 scattering at a selected depth. The
post-plot paragraph must explicitly say that the total scattering array
remains separate from absorption and does not enter the thermal numerator.

### 5.10 An ordered component budget is scientific state

Now assemble named sampled-frequency components before any grid interpolation.
The atmosphere budget follows its explicit IFOP groups. The standard synthesis
sample-column absorption order is:

1. H-minus;
2. H I;
3. H2-plus, He-minus, and compact neutral metals;
4. He I;
5. He II;
6. hot-metal continuum;
7. Si II Peach.

Standard synthesis scattering is H I Rayleigh plus Thomson plus the He I and
H2 Rayleigh terms carried by the minor component.

The atmosphere exposes a 20-entry external IFOP vector. The chapter should
show the complete ownership table only now, after all physical groups have
names. Three details must be visible beside it:

- a branch is on only when its flag is exactly integer `1`;
- the runner default is
  `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]`;
- passing `opacity_flags=None` to the low-level continuum module means twenty
  ones and is therefore not the runner default.

IFOP 9 owns CH, OH, CIA, and the five neutral-metal groups together. IFOP 13
needs the H I ground population constructed under IFOP 4, so H2 Rayleigh is
not independently functional when only IFOP 13 is enabled. IFOP 19 is an
optional, default-off Rosseland surrogate and is not included in the standard
physical sum.

Close Movement II with a compact selected-depth ordered-budget table. The code
retains every named component for parity, but this checkpoint does not add a
seventh quantitative plot.

The movement's coverage lock is:

| physical family | atmosphere standard | synthesis standard | sampled diagnostic / precomputed extension |
| --- | --- | --- | --- |
| H-minus, H I/H II, H2-plus, He-minus | yes | yes | scalar / materialized helpers |
| He I/He II/He III | yes | yes | scalar / materialized alternate helpers |
| neutral C/Mg/Al/Si | full families | compact branches | full scalar / full grid families |
| neutral Fe | full family | no separate standard term | full scalar / full grid family |
| N I/O I | lukewarm group | no | no / explicit grids |
| lukewarm C II | yes | no | no / no |
| Mg II/Ca II | lukewarm group | no | no / explicit grids |
| Si II | generated atmosphere table | packaged Peach table | Peach scalar / grid helpers |
| hot bound-free and ionic free-free | yes | yes | scalar / materialized helper |
| CH/OH/H2 CIA | yes | no | no / no |
| Thomson and H/He/H2 Rayleigh | yes | yes, with different H/H2 policies | scalar / materialized helpers |
| Rosseland surrogate | optional, default off | no | no / no |

This table appears only after the shared physics has been taught. Its purpose
is to prevent omitted or invented terms, not to make the reader memorize
repository layout.

## 5. Movement III — Put the same physical question on two exact grids

Movement question:

> Once the opacity formulae are trustworthy at one frequency, where should
> they be evaluated so that an atmosphere can converge and a requested
> spectrum can be synthesized efficiently?

The answer is not one universal grid. There are two exact consumers.

### 5.11 The atmosphere samples every process directly

The atmosphere builds 30,000 wavelengths:

\[
\lambda_i[{\rm nm}]
=10^{1+10^{-4}(i+s-1)},
\]

with one-based \(i\). The start offset \(s\) depends on effective temperature:

| effective-temperature interval | `start_index` | first wavelength |
| --- | ---: | ---: |
| \(T_{\rm eff}<4500\) K | 11601 | 144.577263387 nm |
| \(4500\le T_{\rm eff}<7250\) K | 9599 | 91.1800865275 nm |
| \(7250\le T_{\rm eff}<13000\) K | 7027 | 50.4312810266 nm |
| \(13000\le T_{\rm eff}<30000\) K | 3577 | 22.7876741143 nm |
| \(T_{\rm eff}\ge30000\) K | 1 | 10.0023028502 nm |

The visible code evaluates both sides of all four boundaries and verifies the
first, interior, and final frequency-weight formulae. A compact numerical
table reports the five regimes and states that the step is constant in
\(\log_{10}\lambda\), not in \(\lambda\); this does not add another plot.

Every enabled atmosphere process is evaluated directly at all 30,000
frequencies. There is no edge-triplet reconstruction in this lane.

### 5.12 Why `njit` and `prange` are safe here

Introduce Numba only after the reader sees the workload: many frequencies,
each requiring the same depth-column calculation.

- `njit` compiles a restricted numerical Python function to machine code.
- `nogil=True` allows compiled work to proceed without the Python interpreter
  lock.
- `cache=True` stores compiled code for later processes.
- `parallel=True` permits Numba to parallelize eligible work.
- `prange` asserts that loop iterations may run independently.

The safe parallel ownership is:

```text
one frequency iteration
  -> reads shared, immutable state and tables
  -> computes one complete depth column
  -> writes only that frequency column
```

No iteration writes another frequency's column. This is why frequency is the
safe `prange` axis for Planck, Gaunt, Karzas, Fe, helium, and lukewarm-metal
kernels. Ordered H-minus interpolation, flag assembly, chunk order, and the
two production continuum calls stay outside the parallel loop.

The visible code compares:

1. a short transparent Python kernel;
2. an `njit` serial kernel;
3. an `njit(parallel=True)` kernel using `prange`.

Report compilation time separately from warm execution, and compare one
thread with the available multi-thread setting. The scientific gate is output
agreement. A performance number is labelled as machine-specific, not promised
as universal. Explain the serial/parallel interpolation dispatch near 8192
queries as an overhead tradeoff, not as different physics.

Name the controls only at this implementation boundary:
`NUMBA_NUM_THREADS` selects the thread count, while `NUMBA_CACHE_DIR` or
`PAYNE_ZERO_NUMBA_CACHE_DIR` selects the persistent cache. A cold compile, a
warm cached call, and a one-thread/many-thread call are three distinct timing
conditions.

The atmosphere route remains CPU NumPy/Numba float64. It does not become a GPU
calculation merely because the next lane uses Torch.

### 5.13 A second atmosphere continuum decides which lines can matter

The atmosphere also owns a smaller reference grid:

- 343 physical wavelength values from 9.09 to 400000 nm;
- a duplicated 344th wavelength and packed sentinel `2**30`;
- an effective-temperature-dependent active subset of 226, 240, 263, 299, or
  338 frequencies.

The first 343 packed coordinates use
`int(log(wavelength_nm) / log(1 + 1/2_000_000) + 0.5)`. Only reference
wavelengths strictly greater than the first active 30,000-grid wavelength are
evaluated.

It performs a second continuum evaluation and constructs a `(depth, 344)`
float32 line-selection threshold,

\[
q_\nu=
\frac{10^{-3}
(\kappa_\nu^{\rm abs}+\kappa_\nu^{\rm sca})}
{\max[1-\exp(-h\nu/kT),10^{-300}]}.
\]

This is not a saved raw continuum. The stimulated-emission factor has been
divided out for a later line-strength comparison. The visible cell verifies
active counts, dtype, the duplicated last column, and the sentinel. It does
not yet teach line selection itself. Inactive short-wavelength entries retain
the exact `1e10` placeholder before the same `1e-3 / stimulated_emission`
scaling; that policy receives a boundary test rather than being silently
replaced by zero.

### 5.14 Synthesis samples only the edge intervals a requested window uses

An arbitrary synthesis wavelength grid may contain many pixels but intersect
only a bounded number of continuum-edge intervals. The exact edge table has
341 wavelength edges and 340 intervals.

For one interval with
\(\lambda_L<\lambda_M<\lambda_R\), evaluate:

\[
\nu_L=\frac{|\nu(\lambda_L)|}{1.0000001},\qquad
\nu_M=\frac{c}{(\lambda_L+\lambda_R)/2},\qquad
\nu_R=|\nu(\lambda_R)|\,1.0000001.
\]

The tiny offsets are directional: the left sample is just to the red side of
the left edge and the right sample is just to the blue side of the right edge.
At an exact internal edge,
`searchsorted(..., side="right") - 1` selects the interval on its red side.
An original schematic must make this one-sided choice visually unambiguous.

The stored edge frequency is signed, but the current sampler takes its
magnitude at both ends. Preserve and checksum the signed field, then verify
that flipping its sign leaves the current sampling result unchanged. Do not
invent an operational sign effect that is absent from the implementation.

For a requested window:

1. find and clip the interval index of every wavelength;
2. retain only unique used intervals;
3. evaluate exactly three physical opacity columns per used interval;
4. interpolate those values onto all requested pixels.

The evaluation count is therefore
`3 * number_of_unique_used_intervals`, bounded by 1020 rather than by the
number of spectral pixels.

The table also packages `continuum_edge_sample_frequency_hz (1020,)`.
The pipeline reads that field, while the structured schema omits it and the
continuum composite recomputes the same triplets. The recomputed vector must be
bitwise equal to the packaged field before recomputation is taught as exact.

### 5.15 Reconstruct positive opacity in logarithmic space

With

\[
d_2=\frac{(\lambda_R-\lambda_L)^2}{2},
\]

derive the three exact Lagrange basis functions:

\[
\begin{aligned}
L_L(\lambda)&=
\frac{(\lambda-\lambda_M)(\lambda-\lambda_R)}{d_2},\\
L_M(\lambda)&=
\frac{2(\lambda_L-\lambda)(\lambda-\lambda_R)}{d_2},\\
L_R(\lambda)&=
\frac{(\lambda-\lambda_L)(\lambda-\lambda_M)}{d_2}.
\end{aligned}
\]

Verify inline that they sum to one and reproduce all three node values.
Then floor the sampled absorption and scattering independently at `1e-30`,
interpolate `log10(opacity)`, and exponentiate.

The distinction must be explicit:

- a component may be physically zero on the inactive side of a threshold;
- the final interpolated slab receives a small positive numerical floor.

One one-panel plot shows the three samples, the exact edge, the reconstructed
curve, and a denser direct diagnostic calculation for context. It should not
combine several edges or several depths.

### 5.16 Host decisions and device arithmetic have different owners

Explain the hybrid boundary after the edge algorithm is visible.

Host float64 owns:

- source tables;
- threshold and bracket searches;
- edge lookup and unique-interval selection;
- edge basis construction;
- H-minus and Si II bracket decisions.

Torch device tensors in the resolved work dtype own populations and the large
opacity columns. Small basis arrays move to the device; full continuum slabs
do not move back to the host at this stage.

The public synthesis policy is:

- CUDA, then MPS, then CPU;
- omitted dtype gives float64 on CUDA/CPU and float32 on MPS;
- requesting float64 on MPS raises `ValueError`.

One deliberate exception must be demonstrated:
`ContinuumTables.from_npz(device="cpu", dtype=None)` uses the module default
float32 because it does not call the public runtime resolver. Reader code
passes the resolved dtype explicitly.

End the movement with the four-lane table. The atmosphere line-reference row
is an explicitly labelled subroute of the first lane:

| lane | grid and algorithm | process-set distinction | output |
| --- | --- | --- | --- |
| atmosphere standard | direct 30,000-point CPU float64 grid | full atmosphere groups, including CH/OH/CIA | absorption, scattering, absorption-weighted source |
| ↳ atmosphere line-reference subroute | 343 physical points plus duplicate/sentinel | second evaluation owned by the atmosphere product | `(D,344)` float32 threshold |
| synthesis standard | only used edge triplets, then log-parabolic interpolation | compact neutral metals; no CH/OH/CIA; no `FrequencyInvariants` | `(D,W)` absorption and scattering |
| sampled diagnostic | caller frequencies, per-frequency fallback, Coulomb layout `True` | richer component experiments | `(D,F)` absorption, scattering, and \(B_\nu\) |
| sampled precomputed extension | explicit `FrequencyInvariants` | materialized full neutral/light-element helpers; not the standard pipeline | independently checked sampled arrays |

`FrequencyInvariants` should be inspected as an available optimization and
extended process object. It must not be inserted into the standard pipeline
or described as a performance-only rewrite: the audited lanes have
route-sensitive formulae and need distinct tests.

## 6. Movement IV — Assemble, expose exact names, and prove the destination

Movement question:

> Can the reader-built physical pieces reproduce the two intended continuum
> consumers without importing the reference project or hiding branch errors
> inside a total?

### 5.17 Bind exact state and table names

Only now present the complete implementation contract. Keep it compact and
grouped by role.

#### Shared state ideas

| physical role | exact fields |
| --- | --- |
| thermodynamic depth vectors | `temperature`, `mass_density`, `electron_density` |
| actual H/He densities | `hydrogen_neutral_population`, optional `hydrogen_ionized_population`, `helium_neutral_population`, `helium_singly_ionized_population` |
| normalized stage bases | `hydrogen_partition_normalized_ion_stage_populations`, `partition_normalized_populations` |
| actual ionic charge partners | `ion_stage_populations` |
| atmosphere-only molecular inputs | packed `ch_population`, `oh_population`, and hydrogen departure coefficients |

Chapter 4's 27-field schema-v4 mapping remains the upstream synthesis
atmosphere. The standard pipeline projects from it an exact trimmed 18-field
continuum view. That synthesis projection is not the packed 18-field
`ContinuumAtmosphereState`. It omits the optional
`hydrogen_ionized_population`, so the H II term uses the second normalized
hydrogen-stage column; it also omits and does not consume
`molecular_hydrogen_population`. One executable shape/projection check must
show both consumer views without repeating Chapter 4's full field tables.

#### Static input roles

Keep distinct, provenance-bound bundles:

- synthesis `continuum_tables.npz` and `continuum_edge_grid.npz`;
- atmosphere `continuum_opacity_tables.npz`,
  `karzas_latter_tables.npz`, and
  `molecular_equilibrium_tables.npz`.

Do not merge atmosphere CH/OH/CIA tables into synthesis
`ContinuumTables`. Do not stage `continuum_level_tables.npz` as an active
dependency: its exported loader is inactive in the audited route.

Every table field has shape, axis, stored numeric convention, physical unit,
consumer, source hash, and role in the manifest. Loader-required but unused
Gavrila/Peach fields in the atmosphere archive are labelled
`required_for_loader`, not `consumed_by_process`.

The exact implementation also has two constant tiers: exact CODATA literals
for physical formulae and rounded reference literals inherited by particular
fits. Present one compact ownership table here. In particular, retain the
atmosphere `WAVENUMBER_PER_EV_REFERENCE` and every route-sensitive continuum
constant where its owning formula uses it. Replacing a rounded fit constant
with a newer-looking exact value is a scientific change, not cleanup.

### 5.18 Run the atmosphere route from named components

The reader-built route should:

1. validate the physical input domain;
2. build the atmosphere continuum state;
3. pass the explicit runner-default IFOP vector;
4. evaluate every named absorption component in exact group and sum order;
5. retain H I and H-minus source numerators before the total can hide them;
6. assemble scattering separately;
7. divide the source numerator by absorption with the exact zero-absorption
   fallback;
8. build the 30,000-point grid and the 343/344-point line threshold;
9. only then open the pinned comparison outputs.

The standard outputs are NumPy float64 arrays
`continuum_absorption`, `continuum_scattering`, and `continuum_source`, each
with shape `(D,30000)`. The line threshold is `(D,344)` float32.

Do not add a final finite/nonnegative repair clamp and call it parity.
Instead, validate positive finite frequency, positive temperature, and
positive mass density before entering exact kernels, then prove the valid
fixtures need no repair.

IFOP 19 is exercised separately for implementation coverage. Its optional
Rosseland-table surrogate uses a bolometric temperature-fourth source
convention rather than ordinary per-frequency \(B_\nu\) units. Preserve that
only in a labelled optional-branch parity test; never mix it into the standard
physical source budget.

### 5.19 Run the standard synthesis route, then label the alternatives

The standard reader-built synthesis route should:

1. use the exact trimmed atmosphere view;
2. call `build_pops(...)` at the implementation boundary;
3. build edge triplets for all intervals and evaluate only those used;
4. preserve the standard `coulomb_table_energy_first=False` layout;
5. preserve the exact sampled-column component order;
6. interpolate absorption and scattering separately in log space;
7. produce `continuum_absorption` and `continuum_scattering`, each `(D,W)`;
8. form the exact downstream
   `continuum_opacity = continuum_absorption + continuum_scattering`;
9. only then compare with the pinned standard-pipeline output.

After this destination is closed, run two clearly labelled secondary checks:

- `compute_sampled_continuum(...)` without invariants as the diagnostic lane;
- the explicitly precomputed `FrequencyInvariants` lane as an implemented
  extension.

Their results must not be presented as the standard route. A call-trace cell
should show that `SynthesisPipeline.run()` reaches `continuum(...)` without
passing `FrequencyInvariants`.

Record a validated wavelength domain for the precomputed extension. Do not
claim global whole-slab equality with the scalar sampled lane: the audited
extension can become non-finite in the far UV and has route-sensitive finite
differences. Shared helpers receive component-level tests; complete extension
slabs receive only domain-bounded claims.

### 5.20 Prove local physics before final slab parity

Use four compact regimes:

- hot dwarf;
- solar dwarf;
- low-gravity giant;
- cool molecule-rich atmosphere.

The order of evidence is:

1. source and static-input hash/shape checks;
2. local dimensional and analytic limits;
3. population perturbation ownership;
4. threshold and interpolation boundaries;
5. named component arrays;
6. ordered partial sums;
7. final atmosphere and synthesis slabs;
8. available backend comparisons with dtype-specific tolerances.

The final four-regime product evidence is a compact numerical parity table,
not a seventh plot or a set of nearly overplotted reference curves. The six
earlier one-claim figures already establish the physical trends; the final
table reports component and slab agreement directly.

Mandatory inline claims include:

- \(n\sigma/\rho\) has cm\(^{2}\) g\(^{-1}\) units;
- the stimulated factor has the correct two limits;
- thresholded components vanish on their inactive side before interpolation
  floors;
- component opacity scales linearly with its owned population and inversely
  with mass density;
- charge-square free-free uses actual ionic populations;
- Thomson is wavelength independent at fixed \(n_e/\rho\);
- Rayleigh rises blueward in a selected valid interval and respects its caps;
- every requested synthesis wavelength maps to one interval;
- recomputed edge triplets match the packaged 1020-sample vector bitwise, and
  the current sampler is invariant to the stored frequency signs;
- the three basis functions reproduce their nodes and sum to one;
- unused edge intervals perform no opacity evaluation;
- atmosphere CH/OH/CIA are present and standard synthesis CH/OH/CIA are
  absent;
- changing stored schema H2 does not change standard synthesis continuum;
- one-thread/many-thread atmosphere results and available synthesis backends
  agree under declared tolerances;
- final valid-domain arrays are finite and nonnegative without a repair clamp.

### 5.21 Chapter summary and next link

The summary should answer the opening question in six short statements:

1. A cross section becomes mass opacity through \(n\sigma/\rho\).
2. Bound-free, free-free, and collision-induced absorption transfer photon
   energy to matter; stimulated emission reduces the net LTE absorption.
3. Thomson and Rayleigh scattering redirect radiation and remain separate from
   the thermal source numerator.
4. Population roles are physical: bound-state terms use normalized bases,
   while ionic free-free terms use actual charge partners.
5. The atmosphere directly samples a 30,000-point CPU grid; synthesis samples
   only used edge triplets and interpolates positive opacity in logarithmic
   space.
6. Exactness means matching each lane to its own process, state, grid, and
   dtype contract—not forcing distinct consumers to equal one another.

The closing link should be causal and brief:

> The smooth background is now accounted for, yet the opening spectrum still
> contains narrow features that no continuum process can make. The next
> chapter gives one bound-bound transition a strength, width, and profile.

## 7. Exact notation reveal map

This map prevents implementation names from appearing before their concepts
are earned.

| concept first earned in | conceptual notation | exact name revealed afterward |
| --- | --- | --- |
| 5.1 | mass density \(\rho\) | `mass_density` |
| 5.1 | one component \(\kappa_{\nu,j}^{\rm abs}\) | local checked value; no aggregate name yet |
| 5.3 | sampled frequency \(\nu\) | `frequencies_hz` |
| 5.4 | scattering mass coefficient | `continuum_scattering` |
| 5.4 | continuum source | `continuum_source` |
| 5.6 | actual electron density | `electron_density` |
| 5.6 | normalized H-stage bases | `hydrogen_partition_normalized_ion_stage_populations` |
| 5.7 | actual/normalized population cubes | `ion_stage_populations`, `partition_normalized_populations` |
| 5.8 | neutral H; normalized CH/OH inputs; local actual H2 | exact H, packed CH/OH slots, and H2 consumer-policy fields |
| 5.10 | ordered absorption sum | `continuum_absorption` |
| 5.11 | atmosphere sample wavelength | atmosphere opacity-grid arrays |
| 5.13 | reference threshold | exact line-reference fields and sentinel |
| 5.14 | requested synthesis wavelength | `wavelength_grid_nm` |
| 5.14 | edge geometry | `signed_continuum_edge_frequency_hz`, `continuum_edge_wavelength_nm`, `continuum_edge_midpoint_wavelength_nm`, `continuum_edge_interval_width_squared_over_two_nm2` |
| 5.16 | device-resident work state | resolved device/dtype and `ContinuumTables` |
| 5.19 | standard public composite | `continuum(...)`, then secondary diagnostic/extension names |

Exact internal routine names appear only in code-cell captions or the compact
route contract after the relevant process has been explained. Reader prose
should say “the H-minus bound-free helper” before it says
`_hminus_bf_scalar`.

## 8. Visible-code plan

Every visible scientific cell has one purpose, stays roughly 10–30 lines, and
is followed immediately by prose that interprets its result. Long exact
kernels live in the progressive textbook package and are called, not pasted
into Markdown or dumped into a cell.

| checkpoint | one live question | visible result |
| ---: | --- | --- |
| 1 | How does one microscopic area become opacity per gram? | unit identity plus population/density scaling |
| 2 | How large can the stimulated-emission correction be? | two limits and one one-panel curve |
| 3 | Are the first physical tables exactly the declared local inputs? | just-in-time H-minus table/manifest identity |
| 4 | Why can H-minus exist without a stored H-minus field, and where does its bound-free branch turn on? | local-factor ownership, edge probes, and one H-minus plot |
| 5 | Why do solar and hot layers have different H/He budgets? | named H/He partial sums and one regime plot |
| 6 | Which population view owns metal bound-free and free-free? | independent normalized/actual perturbation plus named metal sums |
| 7 | Which molecular continua belong to which route? | CH/OH versus local-H2 ownership, one temperature seam, stored-H2 non-use, and one cool-state plot |
| 8 | What does scattering do with wavelength? | Thomson constancy, Rayleigh colour trend, exact caps, and one scattering plot |
| 9 | How does effective temperature choose both atmosphere continuum subroutes? | five starts/weights plus active line-reference count, dtype, duplicate, and sentinel |
| 10 | Why may one frequency column run independently? | Python/serial/parallel equality, cold versus warm timing, thread report |
| 11 | Do the one-sided samples support the promised reconstruction? | exact-edge assignment, used-interval count, basis sum, node reproduction, one reconstruction plot |
| 12 | How does each exact consumer receive only the fields and hardware it owns? | 27→18 synthesis projection, separate atmosphere adapter, and route-specific dtype/device contract |
| 13 | Does the full atmosphere route preserve every component? | named partial sums, source terms, 30,000-point slabs, then atmosphere golden parity |
| 14 | Does the standard synthesis route use only its declared lane? | call trace, final `(D,W)` slabs, synthesis parity, and a compact identity table for the two secondary lanes |

Thirteen to fourteen cells is the target and sixteen is the hard ceiling.
Checkpoints may share one cell when the causal question remains singular.
Hidden setup may load
checksum-bound data and plotting helpers, but it must not precompute the
reader's scientific result.

## 9. Original schematic and one-panel plot plan

All schematics should share the website-inspired visual language while being
new, chapter-specific compositions. They explain dependencies and policies;
they are not screenshots of source or copied website images.

### Schematics

1. **Cross section to mass opacity.** One photon, absorber population, effective
   area, path length, and division by mass density. The final arrow ends at
   cm\(^{2}\) g\(^{-1}\).
2. **Absorption versus redirection.** One incoming photon forks into energy
   deposition and direction change; only the absorptive branch enters the
   thermal numerator.
3. **Frequency-column ownership under `prange`.** Immutable state on the left,
   independent frequency workers in the centre, disjoint depth columns on the
   right. Ordered assembly remains outside the worker boxes.
4. **Two grids, one physical budget.** The atmosphere branch evaluates all
   30,000 samples directly. The synthesis branch selects used intervals,
   enlarges one threshold interval to show its three one-sided samples and
   exact red-side assignment, and interpolates to requested pixels.

### One-panel plots

| plot | sole claim |
| --- | --- |
| stimulated factor | correct low- and high-energy limits |
| H-minus around its edge | bound-free turns on while free-free remains smooth |
| H/He regime comparison | dominant absorbers change between solar and hot layers |
| cool molecular continuum | CH/OH/CIA are real atmosphere contributions in a cool state |
| scattering budget | Thomson is grey and Rayleigh rises blueward |
| edge reconstruction | three one-sided samples recover a positive smooth interval |

The atmosphere grid regimes and final component parity are compact numerical
tables rather than extra plots. This keeps the chapter to six one-panel
figures, each with one scientific claim.

Use the shared professional plotting helper, colourblind-safe paper palette,
units on every axis, restrained legends, and no unexplained normalization.
Avoid multi-panel figures. A second physical claim receives a second figure or
is stated numerically.

## 10. Redundancy and deferral map

### Reuse without rederivation

| topic available to this chapter | minimum self-contained recap | do not repeat |
| --- | --- | --- |
| Planck function | one sentence defining \(B_\nu(T)\) and its unit | blackbody derivation |
| actual versus normalized populations | local definition \(\widetilde n=n/U\) plus one ownership perturbation | Saha equation and partition-sum construction |
| molecular state | CH/OH populations exist; H2 has consumer-specific policies | coupled molecular equilibrium, Newton solver, catalog decoding |
| host/device conventions | host makes discrete float64 choices; device carries large tensors | generic Torch tutorial |
| opacity and source | absorption, scattering, and thermal numerator definitions | full radiative-transfer equation |

### Owned here and never rederived later

- \(n\sigma/\rho\) number-to-mass conversion;
- stimulated-emission factor for continuum opacity;
- H-minus, H I/H II, H2-plus, He-minus, He I/II/III continua;
- neutral, lukewarm, and hot metal continuum families;
- CH/OH photodissociation and H2 CIA as atmosphere-only terms;
- Thomson and H/He/H2 Rayleigh scattering;
- exact atmosphere continuum grids and Numba frequency parallelism;
- exact synthesis edge triplets and log-parabolic opacity interpolation;
- standard/diagnostic/extension route distinctions.

Later chapters may consume continuum slabs or the line-reference threshold.
They should not rederive these processes or interpolation rules.

### Deliberately deferred

| deferred topic | reason and destination |
| --- | --- |
| oscillator strengths and bound-bound transitions | one spectral line is the next causal object |
| Doppler, damping, Voigt profiles | line-shape physics, not continuum |
| line catalog selection inequality and accumulation | consumes the threshold but is a separate routing problem |
| transfer iteration with scattering | needs both continuum and line opacity |
| Rosseland-mean derivation | the optional IFOP 19 surrogate is only parity-tested here, default off |
| atmosphere convergence and blanketing feedback | consumes opacity after the standalone continuum is trusted |
| inverse fitting | outside the atmosphere/synthesis construction scope |

### Material included once, not duplicated across lanes

- Teach each shared microscopic process once, then give a lane-ownership table.
- Explain CH/OH/CIA physics once in Movement II; later mentions are feature
  checks only.
- Explain H2-plus and H2 Rayleigh separately so “H2” does not become one
  ambiguous process.
- Derive the interpolation basis once, in the synthesis lane only.
- Explain Numba once, at the atmosphere frequency-column workload.
- Explain Torch placement once, after the synthesis edge algorithm.
- Keep atmosphere and synthesis parity separate; never repeat the same budget
  merely to force a visual cross-lane comparison.

## 11. Flow and acceptance audit

The draft is ready to implement only if every answer below is “yes.”

### Causal flow

- Does every section begin with a question created by the preceding result?
- Does a physical idea appear before its implementation name?
- Does each plot answer one sentence-sized claim?
- Does each code cell change or verify one conceptual object?
- Does each movement close with a concrete artifact needed by the next?

### Completeness

- Are all atmosphere absorption, scattering, source, flag, H2, grid, Numba,
  and line-reference responsibilities represented?
- Are the standard synthesis compact metals, hot metals, Si II, scattering,
  edge interpolation, host/device split, and output slabs represented?
- Are diagnostic and `FrequencyInvariants` lanes included but labelled
  nonstandard?
- Are CH/OH/CIA presence and absence executable claims?
- Is the optional Rosseland surrogate covered without entering the standard
  sum?

### Nonredundancy

- Is Saha balance absent except for naming its supplied population result?
- Is molecular equilibrium absent except for the exact continuum-local H2
  consumer map?
- Is radiative transfer absent except for defining the source quantity that
  opacity will later supply?
- Is bound-bound *physics* mentioned only in the opening contrast and closing
  handoff, while the exact line-reference threshold is treated only as a
  continuum output?

### Exactness

- Are static inputs opened before goldens, and goldens only after the
  reader-built calculation?
- Is the explicit atmosphere runner-default flag vector passed?
- Is process addition order preserved?
- Are supported physical input preconditions stated instead of applying a
  repair clamp?
- Is the synthesis standard pipeline the primary parity destination?
- Are atmosphere and synthesis checked against their own pinned consumers,
  never against each other by assumption?

### Pedagogy

- Could a reader with basic mathematics and physics define every new
  spectroscopy term in their own words?
- Does the prose interpret every output before moving on?
- Are the exact names discoverable without turning the chapter into a source
  tour?
- Does the summary answer the opening question and explain why one spectral
  line is the only natural next object?
