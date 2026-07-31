# Chapter 5 transparent-teaching-helper contract

Status: binding design contract; no executable implementation is defined here  
Chapter: 5, **Continuous Opacity and Scattering**  
Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

## 1. Purpose

Chapter 5 needs small calculations that a reader can understand line by line,
but its destination is the exact continuum implementation rather than a second,
cleaned-up continuum package. This document fixes the boundary between those
two needs.

Every teaching helper below must do one of three things:

1. expose one physical law with a dimensional or limiting check;
2. expose one exact discrete policy that would be hidden by a final opacity
   slab; or
3. call an already-earned production object and report a narrowly defined
   parity result.

A helper is not a new public interface. A local teaching name may describe a
checkpoint, but it must not rename a production field, routine, route, or
array. Long source kernels remain in the staged canonical modules. Notebook
Markdown never contains copied source blocks.

The helper sequence assumes only the Chapter 4 handoff:

- the 18-field packed `ContinuumAtmosphereState` view for the atmosphere
  consumer; and
- the separate schema-v4 mapping from which the standard synthesis consumer
  receives its trimmed 18-field view.

It does not repeat Chapter 3's ionization calculation, Chapter 4's molecular
equilibrium, or Chapter 6's bound-bound line physics.

## 2. Rules shared by every helper

### 2.1 Reveal order

Each checkpoint follows the same order:

```text
physical question
  -> mathematical quantity and unit
  -> smallest transparent calculation
  -> analytic or limiting check
  -> exact production name, only when earned
  -> exact downstream parity gate
```

Production names may appear earlier in a short reads/writes contract, but the
explanatory prose uses the physical noun until the reader has built the
corresponding object.

### 2.2 Input and output discipline

- Depth is outermost to innermost and has size `D`.
- Atmosphere frequency arrays have size `F`; atmosphere continuum slabs are
  `(D,F)` NumPy `float64`.
- Requested synthesis wavelengths have size `W`; standard synthesis slabs are
  `(D,W)` Torch tensors in the resolved work dtype.
- Frequency is in Hz, wavelength is in nm unless an exact table explicitly
  uses inverse centimetres or Angstrom, temperature is in K, number density is
  in cm\(^{-3}\), mass density is in g cm\(^{-3}\), cross section is in
  cm\(^{2}\), and mass opacity is in cm\(^{2}\) g\(^{-1}\).
- Inputs must be finite and physically positive where the exact formula
  requires it. A helper may not add a final non-negativity or finiteness clamp
  and call the mutation a parity result.
- Static inputs are opened before a reader-built result. Golden outputs are
  opened only afterward.

### 2.3 Evidence discipline

Each helper has two gates.

- The **local gate** is dimensional, analytic, limiting, or an exact discrete
  boundary.
- The **production gate** compares the same constructed quantity with its
  named contribution, route, table, or source output. Distinct production
  lanes are never required to equal one another merely because they describe
  related physics.

Plots are generated from the helper's live outputs. Each plot is one panel
with one physical claim, labelled axes and units, the shared professional
style, and immediate prose interpretation. Conceptual route distinctions use
schematics or compact tables instead of decorative plots.

## 3. Helper inventory and visible-cell budget

The sixteen contracts below fit into fourteen reader-visible cells:

| visible cell | helper contracts | one causal purpose |
| ---: | --- | --- |
| 1 | H01 | turn microscopic interaction probability into opacity per gram |
| 2 | H02 | expose stable net LTE absorption and its two limits |
| 3 | H00 | bind only the physical tables needed by the first table-driven absorber |
| 4 | H03 | derive the local H-minus factor and expose its threshold plus smooth free-free companion |
| 5 | H04 | build the H/He/light-particle budget in two thermal regimes |
| 6 | H05 | prove that metal bound-free and free-free terms own different population views |
| 7 | H06 | expose the atmosphere-only molecular gates and local H2 policy |
| 8 | H07 | keep redirection separate and show its wavelength dependence |
| 9 | H08 and H10 | build the direct atmosphere grid and its line-reference subroute |
| 10 | H09 | prove why a frequency-column loop may use `njit` and `prange` |
| 11 | H11 | reconstruct positive opacity from used edge triplets |
| 12 | compact B1--B4 reads table | execute the 27→18 synthesis projection, contrast the separate atmosphere adapter, and bind device/dtype ownership |
| 13 | B1 | run and check the full atmosphere product |
| 14 | B2, then compact B3--B4 identities | close the standard synthesis product and label the two secondary lanes without promoting them |

H01--H11 are transparent physics or numerical-policy checkpoints. B1--B4 are
the four route/state boundaries. The latter may share display cells, but each
retains an independent test and parity record.

The six planned numerical figures are:

1. stimulated-emission factor;
2. H-minus across its bound-free edge;
3. solar versus hot H/He budget;
4. cool CH/OH/CIA absorption;
5. Thomson and Rayleigh scattering;
6. one synthesis edge reconstruction.

Grid policies, Numba timing, line-reference structure, state reads, and final
parity are clearer as compact tables or schematics.

## 4. Transparent physics and numerical helpers

### H00 — Manifest and table preflight

**Physical question.** Which measured or fitted cross sections are about to
enter the calculation, and how do we know they are the intended arrays rather
than similarly named substitutes?

**Equation.**

\[
{\rm SHA256}(\mathrm{file\ bytes}) =
{\rm digest}_{\rm manifest}.
\]

The equality is a scientific identity condition because changing a table
changes \(\kappa_\nu\), even when all Python formulae remain unchanged.

**Exact production name, when earned.** After the reader understands that a
cross section is a physical input, reveal
`ContinuumOpacityTables`, `KarzasLatterTables`,
`MolecularEquilibriumTables`, and synthesis `ContinuumTables`, together with
`load_continuum_opacity_tables`, `load_karzas_latter_tables`,
`load_molecular_equilibrium_tables`, and `ContinuumTables.from_npz`.
`ContinuumLevelTables` and `load_continuum_level_tables` remain explicitly
inactive.

**Inputs, units, and shapes.**

- staged source modules and their SHA-256 values;
- atmosphere `continuum_opacity_tables.npz`,
  `karzas_latter_tables.npz`, and `molecular_equilibrium_tables.npz`;
- synthesis `continuum_tables.npz` and `continuum_edge_grid.npz`;
- manifest member metadata: exact key, shape, dtype, stored unit or numerical
  convention, source identity, and consumer role.

The just-in-time visible preflight includes the H-minus `(85,)` wavelength and
stored cross-section arrays and the `(22,)` by `(11,)` free-free axes needed
by H03. Later checkpoints verify the Karzas-Latter `(29,15)` total grids, the
`(200,)` atmosphere H2 partition table, and the 341-edge/340-interval
synthesis geometry immediately before their first use. One compact final
identity table confirms that all five immutable archives passed; no opening
cell front-loads the complete implementation inventory.

**Smallest honest implementation.** Hash each immutable archive before its
first table-driven consumer, compare its digest with the manifest, load it
through the exact loader, and print only the fields used by the next live
question. Do not dump an archive or paste a loader. The first such call occurs
after H01--H02 and immediately before H03.

**Analytic or limiting check.** Deliberately compare one altered byte string
with the manifest digest in memory and show that identity fails. Do not alter
the checked-in file.

**Intended plot.** None.

**Exact downstream production parity gate.** Every subsequent helper receives
the already-preflighted table objects. Tests require exact module hashes,
archive hashes, field names, shapes, dtypes, and active/inactive roles before
any golden output can be opened.

### H01 — Cross section to mass opacity

**Physical question.** How does an effective interaction area for one particle
become an extinction coefficient for one gram of gas?

**Equation.**

\[
\alpha_\nu=n_{\rm absorber}\sigma_\nu
\quad[{\rm cm}^{-1}],
\qquad
\kappa_\nu=\frac{\alpha_\nu}{\rho}
=\frac{n_{\rm absorber}\sigma_\nu}{\rho}
\quad[{\rm cm}^{2}\,{\rm g}^{-1}].
\]

The helper performs the unit cancellation in the main text.

**Exact production name, when earned.** There is no production routine for
this algebra alone. Retain the existing teaching names
`mass_opacity_from_cross_section` and `opacity_scaling_checkpoint`; introduce
the exact aggregate name `continuum_absorption` only after named components
are summed in H05.

**Inputs, units, and shapes.**

- `absorber_number_density_cm3`: scalar or broadcast-compatible array,
  cm\(^{-3}\);
- `cross_section_cm2`: scalar or broadcast-compatible array, cm\(^{2}\);
- `mass_density_g_cm3`: scalar or broadcast-compatible array,
  g cm\(^{-3}\);
- output: the broadcast shape, NumPy `float64`, cm\(^{2}\) g\(^{-1}\).

**Smallest honest implementation.** Validate finite positive inputs, use one
broadcasted multiplication and division, and return the physical value without
a repair clamp.

**Analytic or limiting check.** Doubling absorber population doubles opacity;
doubling mass density halves opacity; a hand-computed scalar equals
`1.5e4 cm2 g-1` for the existing checkpoint values.

**Intended plot.** None; the factor-of-two table is more direct.

**Exact downstream production parity gate.** For one isolated process with all
other owned state fixed, perturb its population by two and its mass density by
two. The named production component must follow the same factors wherever its
cross section and population are otherwise independent of those perturbations.
Departure-coefficient and chemistry-dependent cases are excluded from this
simple scaling claim and tested at their own boundaries.

### H02 — Stable stimulated emission

**Physical question.** Why is gross absorption reduced when the same radiation
can stimulate the reverse transition, and how can the small difference be
computed without cancellation?

**Equation.**

\[
x=\frac{h\nu}{kT},
\qquad
s_\nu(T)=1-\exp(-x)=-\operatorname{expm1}(-x).
\]

Its limits are

\[
s_\nu \sim x\quad(x\ll1),\qquad
s_\nu\to1\quad(x\gg1).
\]

**Exact production name, when earned.** Retain
`stimulated_emission_factor` and `stimulated_emission_checkpoint` as
teaching helpers. The production component names are not revealed here because
the exact source decides whether a fitted coefficient already contains the
equivalent convention.

**Inputs, units, and shapes.**

- `frequency_hz`: positive scalar or array, Hz;
- `temperature_k`: positive broadcast-compatible scalar or array, K;
- factor: broadcast shape, NumPy `float64`, dimensionless;
- the plot uses dimensionless \(x=h\nu/kT\), spanning the low- and high-energy
  limits.

**Smallest honest implementation.** Compute `-expm1(-x)` after validating and
broadcasting the two physical inputs. Do not subtract `exp(-x)` from one in
the teaching implementation.

**Analytic or limiting check.** Require \(0<s_\nu\le1\), agreement with \(x\)
at small \(x\), and approach to one at large \(x\). Include one ordinary
photospheric frequency/temperature pair.

**Intended one-panel plot.** \(s_\nu\) against \(x=h\nu/kT\), with the
low-energy \(s=x\) guide and high-energy \(s=1\) guide. The sole claim is that
one stable formula reaches both limits.

**Exact downstream production parity gate.** At component level, verify the
factor at the frequencies and temperatures consumed by H-minus, H I, metals,
and the line-reference threshold. The gate is formula ownership, not a blind
second multiplication: each exact source term must match its pinned
stimulated-emission convention once and only once.

### H03 — H-minus threshold and smooth free-free absorption

**Physical question.** How can one absorber create a sharp bound-free onset
and a smooth free-free background at the same depth?

**Equation.**

For the table-driven bound-free term,

\[
\sigma_{\nu}^{\rm H^-,bf}=0
\quad\hbox{for}\quad
\nu\le\nu_0,\qquad
\nu_0=1.82365\times10^{14}\ {\rm Hz},
\]

and, in the LTE controlled limit,

\[
\kappa_\nu^{\rm H^-,bf}
=10^{-18}\sigma_{\rm stored}(\lambda)
\frac{n_{\rm H^-}}{\rho}s_\nu.
\]

The exact atmosphere route generalizes the net factor and source with H-minus
and H I departure coefficients. The free-free term is a separate
temperature/wavelength table and has no corresponding bound-free edge.

**Exact production name, when earned.** After the threshold experiment,
introduce atmosphere `compute_hminus_opacity_columns`. The synthesis internal
`_hminus_opacity` may appear in a code-cell caption or route trace, but it is
not presented as a public teaching API.

**Inputs, units, and shapes.**

- one selected depth from the solar fixture;
- three exact edge probes: immediately below, exactly at, and immediately
  above `1.82365e14 Hz`, plus a small plotting grid;
- `temperature`, K; `mass_density`, g cm\(^{-3}\);
  `electron_density`, cm\(^{-3}\);
  `hydrogen_partition_normalized_ion_stage_populations[:,0]`,
  cm\(^{-3}\) per partition;
- H-minus bound-free wavelength and stored cross-section arrays, both `(85,)`;
  the stored cross-section numbers are in units of
  \(10^{-18}\) cm\(^{2}\), despite the field spelling;
- output: named bound-free, free-free, and ordered-sum opacity,
  `(D,F)` or selected `(F,)`, cm\(^{2}\) g\(^{-1}\).

**Smallest honest implementation.** Interpolate the physical table at the
probe wavelengths, apply the explicit `1e-18`, compute the supplied LTE
population factor, and keep bound-free and free-free arrays separate before
adding them. The long theta-table setup stays in the canonical package.

**Analytic or limiting check.** Bound-free is exactly zero below and at the
strict production threshold and is active just above it. Free-free remains
finite and smooth across the same probes. In the LTE limit the net factor
agrees with H02.

**Intended one-panel plot.** Bound-free, free-free, and their sum against
wavelength near the edge. The sole claim is thresholded plus smooth opacity,
not a complete continuum.

**Exact downstream production parity gate.** Compare the transparent edge
probes and ordered sum with `compute_hminus_opacity_columns` under unit
departure coefficients, then separately compare the fixture's exact departure
state. The synthesis standard component is checked through its named
component capture before the final `continuum` slab; cross-lane equality is
not assumed.

### H04 — H, H2-plus, and helium across two thermal regimes

**Physical question.** Why does a solar-like layer and a hot layer build a
different smooth background even before metals are added?

**Equation.**

A bound level is recovered from a partition-normalized stage population:

\[
n_i=\widetilde n_r\,g_i\exp(-E_i/kT),
\qquad
\widetilde n_r=n_r/U_r(T).
\]

Its LTE bound-free contribution has the pattern

\[
\kappa_{\nu,i}^{\rm bf}
=\frac{n_i\sigma_{\nu,i}}{\rho}s_\nu,
\]

while Coulomb free-free absorption has the ownership pattern

\[
\kappa_\nu^{\rm ff}
\propto
\frac{n_e\,n_{\rm ion}\,Z^2\,g_{\rm ff}}
{\rho\,T^{1/2}\nu^3}s_\nu .
\]

H2-plus and He-minus use their pinned fitted laws rather than pretending to be
new instances of either displayed approximation.

**Exact production names, when earned.** Reveal
`compute_hydrogen_opacity_columns`,
`compute_molecular_hydrogen_ion_opacity_columns`,
`compute_heminus_opacity_columns`,
`compute_helium_neutral_opacity_columns`,
`compute_helium_ionized_opacity_columns`, and finally
`compute_light_element_continuum_columns`.

**Inputs, units, and shapes.**

- solar-like and hot-dwarf atmosphere fixtures, each `D=6`;
- a compact declared frequency grid `(F,)`, Hz;
- actual H/He populations, normalized H/He stage populations, electron
  density, mass density, temperature, and `(D,6)` hydrogen departure
  coefficients with the units fixed in the 18-field atmosphere contract;
- Karzas-Latter tables and Coulomb Gaunt table;
- each component and partial sum: `(D,F)` NumPy `float64`,
  cm\(^{2}\) g\(^{-1}\); source arrays are \(B_\nu\) or the exact
  departure-corrected source in
  erg s\(^{-1}\) cm\(^{-2}\) sr\(^{-1}\) Hz\(^{-1}\).

**Smallest honest implementation.** Construct one H I bound level
transparently. Then call the short exact component checkpoints for the
explicit H ladder and tail, H II free-free, H2-plus, He-minus, He I, and
He II/III. Preserve named arrays and exact addition order; do not reproduce
the large shell and autoionization kernels in a notebook cell.

**Analytic or limiting check.** The one-level population decreases with
excitation energy at fixed \(T\); bound-free terms vanish below their
thresholds; free-free ownership changes with actual ion/electron density.
The two fixtures must show an interpretable regime change rather than merely
two nonzero totals.

**Intended one-panel plot.** A restrained solar-versus-hot comparison with a
small number of grouped H/He curves. The sole claim is that the dominant
light-particle background changes with thermal/ionization state.

**Exact downstream production parity gate.** Named component arrays must sum
in pinned order to `compute_light_element_continuum_columns`; the source
numerator reconstructed from H I, H-minus, and LTE thermal components must
reproduce its returned source. The same physical families are checked
separately in the standard synthesis component capture.

### H05 — Neutral and ionized metal ownership

**Physical question.** Which metal populations supply bound electrons, and
which supply the charged partners for free-free absorption?

**Equation.**

\[
\kappa_\nu^{\rm metal,bf}
=\frac{1}{\rho}
\sum_{r,i}
\widetilde n_r\,g_i e^{-E_i/kT}
\sigma_{\nu,i}s_\nu ,
\]

\[
Q_Z=\sum_r q_r^2 n_r,
\qquad
\kappa_\nu^{\rm metal,ff}
\propto
\frac{n_e}{\rho T^{1/2}\nu^3}
\sum_Z Q_Z g_{\rm ff}(q,T,\nu)s_\nu .
\]

The first equation consumes partition-normalized populations; the second
consumes actual ion-stage populations.

**Exact production names, when earned.** Reveal the atmosphere component
names:

- `compute_carbon_neutral_opacity_columns`;
- `compute_magnesium_neutral_opacity_columns`;
- `compute_aluminum_neutral_opacity_columns`;
- `compute_silicon_neutral_opacity_columns`;
- `compute_iron_neutral_opacity_columns`;
- `compute_lukewarm_metal_opacity_columns`;
- `compute_hot_metal_opacity_columns`.

At the synthesis state boundary, `build_pops` earns its exact
`hot_metal_populations` and `charge_square_population_sum` outputs. Compact
standard-synthesis neutral-metal terms remain distinguished from the fuller
atmosphere and sampled-extension families.

**Inputs, units, and shapes.**

- one solar-like and one hot fixture;
- `partition_normalized_populations_by_packed_slot` `(D,1006)` and
  `ion_stage_populations_by_packed_slot` `(D,1006)` for atmosphere;
- `partition_normalized_populations` and `ion_stage_populations`
  `(D,6,139)` for synthesis;
- frequency `(F,)`, Hz, and the exact metal tables;
- named outputs `(D,F)`, float64 atmosphere or resolved synthesis dtype,
  cm\(^{2}\) g\(^{-1}\).

**Smallest honest implementation.** Evaluate one thresholded neutral-metal
level from \(\widetilde n_r\). Then perform two controlled copies of the same
fixture: double only one normalized population, and double only one actual
ion-stage population. These are interface-sensitivity diagnostics, not
chemically self-consistent atmospheres. Use exact component checkpoints for
the remaining neutral, lukewarm, hot, and Si II work.

**Analytic or limiting check.** The normalized-only perturbation changes its
owned bound-free contribution but not the charge-square sum. The actual-only
perturbation changes the corresponding free-free contribution but not the
bound-free base. Thresholded branches vanish on their inactive side. The
60-record hot bound-free accumulation retains its running one-percent
acceptance rule and source order.

**Intended plot.** No additional plot. A compact perturbation table and the
later ordered component-budget table make the ownership claim more clearly.

**Exact downstream production parity gate.** Require each named atmosphere
metal component, its ordered IFOP partial sum, and the synthesis
`build_pops` derived arrays to match the pinned source. Standard synthesis is
checked against its compact-metal component capture; the fuller neutral and
light-element grids belong only to the sampled extension and are not added to
`continuum(...)`.

### H06 — CH, OH, and H2 collision-induced gates

**Physical question.** How can molecules absorb continuously without
re-solving the molecular equilibrium, and why do the two continuum consumers
not read one universal H2 field?

**Equation.**

The packed CH and OH inputs are partition-normalized:

\[
\kappa_\nu^{\rm CH}
=\frac{(N_{\rm CH}/U_{\rm CH})
[\sigma_\nu^{\rm CH}U_{\rm CH}]}
{\rho}s_\nu ,
\]

with the analogous OH expression. The atmosphere reconstructs a local actual
H2 density,

\[
n_{\rm H_2}
=\left(2b_1\widetilde n_{\rm H\,I}\right)^2 K_{\rm H_2}(T),
\]

and uses it in collision-induced absorption,

\[
\kappa_\nu^{\rm CIA}
=\frac{n_{\rm H_2}}{\rho}
\left[
C_{\rm H_2-He}n_{\rm He}
+C_{\rm H_2-H_2}n_{\rm H_2}
\right]s_\nu .
\]

**Exact production names, when earned.** Reveal
`compute_molecular_hydrogen_population` and
`compute_molecular_continuum_opacity_columns`. The exact adapter fields
`ch_population` and `oh_population` are identified as aliases of normalized
packed slots 845 and 847, not generic actual molecular populations.

**Inputs, units, and shapes.**

- cool-molecule-rich and mixed warm/cool atmosphere fixtures;
- `temperature` `(D,)`, K;
- normalized H I ground base and departure coefficient `(D,)`;
- `helium_neutral_population`, `ch_population`, and `oh_population` `(D,)`,
  cm\(^{-3}\) in their declared representation;
- mass density `(D,)`, g cm\(^{-3}\);
- caller frequency `(F,)`, Hz, including the
  20,000 cm\(^{-1}\) CIA boundary;
- CH/OH `(energy,temperature)` tables, collision `(81,7)` tables, and
  H2 partition `(200,)`;
- named absorption outputs `(D,F)`, NumPy `float64`,
  cm\(^{2}\) g\(^{-1}\).

**Smallest honest implementation.** Call the exact local H2 reconstruction,
then retain CH, OH, H2-H2, and H2-He contributions separately in one boundary
checkpoint. Do not rederive mass action or expose the Chapter 4 Newton solve.

**Analytic or limiting check.**

- the molecular composite is entered only when `min(temperature) < 9000 K`;
- CH and OH are zero per layer at and above 9000 K;
- the H2 equilibrium interpolation tests 100/101 K and 19899/19900 K;
- H2 is retained at exactly 20000 K and zero only above it;
- CIA is active at wavenumber at or below 20000 cm\(^{-1}\);
- the reversed-looking lower/upper CIA temperature weights are checked at a
  lower node, midpoint, and adjacent upper node;
- an all-warm column and a mixed cool/warm column test the global gate;
- molecule-disabled input zeroes CH/OH but does not disable the local H2
  reconstruction.

**Intended one-panel plot.** Cool-state CH, OH, and CIA absorption in one
infrared-focused window. The sole claim is that these are real
atmosphere-product continuum components.

**Exact downstream production parity gate.** The four named molecular
contributions and their sum reproduce
`compute_molecular_continuum_opacity_columns`. A standard synthesis call trace
must prove the absence of CH, OH, and CIA. Perturbing schema-v4
`molecular_hydrogen_population` must leave the standard synthesis continuum
unchanged.

### H07 — Thomson and Rayleigh scattering

**Physical question.** How can a photon be removed from one ray without
depositing its energy as heat, and why is that redirection almost grey for
electrons but strongly coloured for neutral particles?

**Equation.**

\[
\kappa_{\rm e}^{\rm sca}
=\sigma_{\rm T}\frac{n_e}{\rho},
\qquad
\sigma_{\rm T}=0.6653\times10^{-24}\ {\rm cm}^{2}.
\]

Rayleigh terms retain the same population-to-mass structure,

\[
\kappa_{\nu,j}^{\rm Ray}
=\frac{n_j\sigma_{\nu,j}^{\rm Ray}}{\rho},
\]

with an approximate \(\lambda^{-4}\) trend away from resonances. The exact
fits, caps, and H2 reconstruction belong to the named route.

**Exact production name, when earned.** Reveal
`compute_continuum_scattering_columns`, then the aggregate
`continuum_scattering`. Synthesis `_scattering_opacity` and the minor
He I/H2 Rayleigh terms remain internal component labels, not a replacement
public API.

**Inputs, units, and shapes.**

- one selected depth and a compact `(F,)` frequency grid;
- electron density, actual neutral H and He populations, local H2 population,
  and mass density;
- atmosphere caps:
  `2.463e15 Hz` for H I, `5.15e15 Hz` for He I, and
  `2.922e15 Hz` for H2;
- outputs: electron, H I, He I, H2, and total scattering,
  `(D,F)` or selected `(F,)`, cm\(^{2}\) g\(^{-1}\).

**Smallest honest implementation.** Compute Thomson directly. Call the exact
Rayleigh component checkpoint for the three neutral scatterers and retain
each named contribution before the sum.

**Analytic or limiting check.** Thomson is wavelength independent at fixed
\(n_e/\rho\). Each Rayleigh contribution rises blueward in a declared valid
interval and flattens after its exact capped input frequency. Atmosphere H2
Rayleigh is present only when both IFOP 4 has constructed the H I ground
population and IFOP 13 is enabled.

**Intended one-panel plot.** Electron, H I, He I, and H2 scattering against
wavelength at one depth. The sole claim is grey versus coloured redirection.

**Exact downstream production parity gate.** Named terms sum exactly in the
pinned order to `compute_continuum_scattering_columns`. The result remains
separate from absorption and never enters the atmosphere thermal-source
numerator. The standard synthesis scattering capture separately checks
Thomson, H I Rayleigh, He I Rayleigh, and its synthesis-specific H2 Rayleigh
policy.

### H08 — Five exact direct atmosphere grids

**Physical question.** Where must the atmosphere evaluate continuum opacity
directly so that a cool star does not spend its fixed sample budget on the
same far-UV range as a hot star?

**Equation.**

\[
\lambda_i[{\rm nm}]
=10^{1+10^{-4}(i+s-1)},\qquad i=1,\ldots,30000,
\]

where \(s\) is the exact effective-temperature-dependent `start_index`.
With \(\nu_i=c/\lambda_i\), the quadrature weights are

\[
\begin{aligned}
w_0&=1.5(\nu_0-\nu_1),\\
w_i&=\tfrac12(\nu_{i-1}-\nu_{i+1}),\\
w_{N-1}&=\tfrac14(\nu_{N-2}+\nu_{N-1}).
\end{aligned}
\]

**Exact production name, when earned.** Reveal
`build_opacity_sampling_grid`; retain
`atmosphere_grid_checkpoint` as the teaching checkpoint.

**Inputs, units, and shapes.**

- `effective_temperature`, scalar K;
- probes on both sides of 4500, 7250, 13000, and 30000 K;
- exact start indices `11601`, `9599`, `7027`, `3577`, and `1`;
- returned wavelength and frequency-weight arrays `(30000,)`,
  NumPy `float64`, in nm and Hz respectively.

**Smallest honest implementation.** Recreate only the three-line logarithmic
grid formula and the three weight cases. Compare it with the exact builder for
all five regimes. A table reports the branch, first wavelength, last
wavelength, sample count, and representative weights.

**Analytic or limiting check.** Every branch has 30,000 points, wavelengths
increase monotonically, the step is constant in \(\log_{10}\lambda\), and the
three weight formulae are positive. Exact boundary assignment is tested on
both sides, not only at the hotter-side equality.

**Intended plot.** None. The two-grid schematic and five-row numerical table
carry the policy without adding a seventh figure.

**Exact downstream production parity gate.** Rebuilt arrays match
`build_opacity_sampling_grid` under the source-appropriate exact/tolerance
policy for every branch. B1 must evaluate the complete atmosphere continuum
at these returned 30,000 frequencies; it may not substitute synthesis edge
triplets.

### H09 — `njit`, `prange`, and honest timing

**Physical question.** Which part of the 30,000-frequency workload is
independent, and what does compilation change without changing the physics?

**Equation.**

For a transparent column kernel,

\[
K_{d f}=\frac{n_d\sigma_f}{\rho_d}.
\]

For fixed \(f\), the worker reads shared immutable depth state and writes only
column \(K_{:,f}\). Therefore

\[
\{K_{:,f}\}\cap\{K_{:,f'}\}=\varnothing
\quad\hbox{for}\quad f\ne f'
\]

as write sets, even though the workers share read-only inputs.

**Exact production name, when earned.** `njit`, `nogil=True`, `cache=True`,
`parallel=True`, and `prange` are explained as distinct decisions. The
production destination remains `compute_continuum_opacity_columns`; private
compiled kernels may appear in a route trace but do not become reader APIs.

**Inputs, units, and shapes.**

- small positive depth vectors `n` and `rho`, each `(D,)`;
- cross sections `(F,)`, cm\(^{2}\);
- output `(D,F)`, cm\(^{2}\) g\(^{-1}\), NumPy `float64`;
- separate subprocess records for Python, serial `njit`, parallel `prange`,
  one thread, available many threads, cold compilation, warm same-process
  call, and fresh-process cache reuse.

**Smallest honest implementation.** Express the same two-loop calculation in
plain Python, serial cached `njit`, and `njit(parallel=True)` with the outer
frequency loop written as `prange`. Each frequency worker owns one complete
depth column. Ordered component assembly remains outside the parallel loop.

**Analytic or limiting check.** All three outputs match the vectorized H01
identity. One-thread and many-thread results satisfy the declared source
tolerance. Timing is reported only after parity and includes machine, thread
count, problem shape, and cache condition.

**Intended plot.** None. A small timing table distinguishes compile time, warm
execution, and fresh-process cache reuse. A conceptual schematic shows
disjoint frequency-column ownership.

**Exact downstream production parity gate.** Run B1 in two truly fresh,
initially empty Numba cache directories, then compare cold and warm results.
Compare fixed one-thread and declared many-thread atmosphere products. Do not
promise universal speedup or bit identity when the exact source tolerance
permits libm or grouping differences. The separate 8192-query serial/parallel
interpolation dispatch is reported as an overhead policy, not different
physics.

### H10 — Atmosphere continuum reference for later line selection

**Physical question.** How can the atmosphere compress its continuum
background into the exact reference scale that a later line-selection stage
will consume?

**Equation.**

\[
q_\nu=
\frac{10^{-3}
(\kappa_\nu^{\rm abs}+\kappa_\nu^{\rm sca})}
{\max[1-\exp(-h\nu/kT),10^{-300}]}.
\]

The division removes the stimulated-emission factor from the comparison
scale. It is not a second definition of continuum opacity.

**Exact production names, when earned.** Reveal
`build_continuum_reference_wavelength_grid`,
`active_continuum_reference_frequencies`, and
`assemble_continuum_line_selection_threshold`.

**Inputs, units, and shapes.**

- 343 physical reference wavelengths from 9.09 to 400000 nm;
- a 344th duplicated wavelength and packed sentinel `2**30`;
- `effective_temperature`, K, and `temperature` `(D,)`, K;
- exact active continuum absorption and scattering `(D,A)`,
  cm\(^{2}\) g\(^{-1}\);
- output threshold `(D,344)` NumPy `float32`, with the pinned comparison-scale
  convention; reference wavelength `(344,)` nm and packed index `(344,)`
  integer.

**Smallest honest implementation.** Build the grid, select wavelengths
strictly greater than the first active 30,000-grid wavelength, evaluate the
exact continuum only at those `A` frequencies, and pass the arrays to the
exact threshold assembler.

**Analytic or limiting check.**

- active counts are 226, 240, 263, 299, and 338 in successively hotter grid
  regimes;
- equality with the first direct-grid wavelength is inactive because the
  comparison is strict;
- the final physical column is duplicated into column 343;
- the last packed coordinate is `2**30`;
- inactive short-wavelength entries begin from the exact `1e10` placeholder
  before the same `1e-3 / stimulated_emission` scaling;
- the returned threshold dtype is exactly `float32`.

**Intended plot.** None. A five-row active-count table and a compact last-two-
column printout are sufficient.

**Exact downstream production parity gate.** Recompute active absorption and
scattering through the atmosphere product subroute and require the assembled
`(D,344)` array, duplicate, sentinel, active counts, and dtype to match the
pinned output. This remains a subroute of B1, not a fifth continuum lane, and
the chapter does not teach the Chapter 6/7 line-selection inequality here.

### H11 — One-sided synthesis edge triplets and log-parabolic reconstruction

**Physical question.** How can synthesis evaluate continuum physics only in
the edge intervals used by a requested wavelength window while preserving
threshold sides and positive opacity?

**Equation.**

For interval
\(\lambda_L<\lambda_M<\lambda_R\),

\[
\nu_L=\frac{|\nu(\lambda_L)|}{1.0000001},\quad
\nu_M=\frac{c}{(\lambda_L+\lambda_R)/2},\quad
\nu_R=|\nu(\lambda_R)|\,1.0000001.
\]

With \(d_2=(\lambda_R-\lambda_L)^2/2\),

\[
\begin{aligned}
L_L&=\frac{(\lambda-\lambda_M)(\lambda-\lambda_R)}{d_2},\\
L_M&=\frac{2(\lambda_L-\lambda)(\lambda-\lambda_R)}{d_2},\\
L_R&=\frac{(\lambda-\lambda_L)(\lambda-\lambda_M)}{d_2}.
\end{aligned}
\]

For either absorption or scattering,

\[
\log_{10}\kappa(\lambda)
=L_L\log_{10}\max(\kappa_L,10^{-30})
+L_M\log_{10}\max(\kappa_M,10^{-30})
+L_R\log_{10}\max(\kappa_R,10^{-30}).
\]

**Exact production names, when earned.** Reveal
`build_edge_sample_frequencies`, then standard synthesis `continuum`.
Retain `edge_triplet_checkpoint` as the teaching checkpoint.

**Inputs, units, and shapes.**

- signed edge frequency `(341,)`, Hz;
- edge wavelength `(341,)`, nm;
- midpoint and half-width-squared arrays `(340,)`, nm and nm\(^{2}\);
- packaged sample frequency `(1020,)`, Hz;
- requested synthesis wavelengths `(W,)`, nm;
- three samples for each unique used interval;
- interpolated absorption and scattering `(D,W)`, resolved Torch dtype,
  cm\(^{2}\) g\(^{-1}\).

**Smallest honest implementation.** Use host `float64` to find
`searchsorted(..., side="right") - 1`, unique used intervals, and the three
basis arrays. Evaluate one declared interval transparently, interpolate its
three positive sample values in log space, and call the exact standard route
for the full requested grid.

**Analytic or limiting check.**

- an exact internal edge is assigned to the interval on its red-wavelength
  side;
- reconstructed sample frequencies match the packaged 1020-vector bitwise;
- flipping stored frequency signs leaves the current triplets unchanged;
- the evaluation count is exactly `3 * number_of_unique_used_intervals`;
- \(L_L+L_M+L_R=1\);
- the basis has the Kronecker-delta values at all three nodes;
- interpolation reproduces all three input log opacities at their nodes;
- a trace proves that unused intervals invoke no opacity evaluation;
- a physically zero component may remain zero before sampling, while only the
  final interpolation inputs receive the `1e-30` floor.

**Intended one-panel plot.** Three one-sided samples, the physical edge, the
log-parabolic reconstruction, and a dense direct diagnostic curve for one
depth and one interval. The sole claim is faithful positive reconstruction in
that interval.

**Exact downstream production parity gate.** Require edge assignment, used
intervals, sample indices, basis arrays, and reconstructed absorption and
scattering to reproduce the internal standard `continuum` route. The final
slabs are then checked in B2. The atmosphere route is never compared to this
interpolation because it evaluates its own 30,000 points directly.

## 5. Four exact route/state boundary helpers

These are four independent checkpoints, not four implementations of one
universal continuum function.

### B1 — Atmosphere product boundary

**Physical question.** Do the named atmosphere components, their flags, and
their source ownership assemble into the exact direct continuum product on
the grid that the atmosphere actually uses?

**Equation.**

\[
\kappa_\nu^{\rm abs}
=\sum_j\kappa_{\nu,j}^{\rm abs},
\qquad
N_\nu=\sum_j\kappa_{\nu,j}^{\rm abs}S_{\nu,j},
\qquad
S_\nu^{\rm cont}=
\begin{cases}
N_\nu/\kappa_\nu^{\rm abs},&\kappa_\nu^{\rm abs}>0,\\
B_\nu,&\kappa_\nu^{\rm abs}=0.
\end{cases}
\]

Scattering is assembled separately:

\[
\kappa_\nu^{\rm sca}=\sum_j\kappa_{\nu,j}^{\rm sca}.
\]

**Exact production names, when earned.** Reveal
`ContinuumAtmosphereState`,
`build_continuum_atmosphere_state`,
`compute_continuum_opacity_columns`, and
`build_opacity_sampling_grid`.

**Inputs, units, and shapes.**

The exact 18-field `ContinuumAtmosphereState`:

- five thermodynamic/velocity `(D,)` vectors;
- hydrogen normalized stages `(D,2)`;
- four actual named H/He `(D,)` populations;
- two normalized named He `(D,)` populations;
- elemental abundances `(D,99)`;
- hydrogen departure coefficients `(D,6)`;
- actual and normalized packed populations `(D,1006)`;
- normalized CH/OH aliases `(D,)`.

The full grid has `F=30000`; outputs are NumPy `float64`
`continuum_absorption`, `continuum_scattering`, and `continuum_source`, each
`(D,30000)`, with opacity in cm\(^{2}\) g\(^{-1}\) and source in
erg s\(^{-1}\) cm\(^{-2}\) sr\(^{-1}\) Hz\(^{-1}\).

**Smallest honest implementation.** Validate the 18 fields, build the exact
grid, pass the explicit runner vector
`[1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,0]`, retain named absorption,
scattering, and source-numerator components in exact order, and only then call
the exact composite.

**Analytic or limiting check.** Every valid component and final slab is finite
and nonnegative without a repair clamp. Scattering does not change the source
numerator. IFOP flags activate only when exactly integer `1`; `None` is
explicitly shown to mean twenty ones and therefore not the runner default.
The reconstructed source residual is checked componentwise before a global
maximum can hide a small term.

**Intended plot.** None at this boundary. Named component and parity tables
avoid repeating earlier physics plots.

**Exact downstream production parity gate.** For hot dwarf, solar dwarf,
low-gravity giant, and cool molecule-rich fixtures, compare named components,
ordered partial sums, all three full `(D,30000)` outputs, the line-reference
subroute, one-thread/many-thread results, and cold/warm results with the pinned
atmosphere oracle. The optional IFOP-19 Rosseland surrogate is a separate
test-only branch and never enters this standard sum.

**Existing-helper limitation to preserve.**
`run_atmosphere_continuum(regime, frequency_hz)` evaluates the exact composite
on a compact caller grid. It is useful for H03--H07 and local route checks, but
it does **not** satisfy this full 30,000-point product boundary.

### B2 — Standard synthesis product boundary

**Physical question.** Do the exact trimmed synthesis state, used edge
triplets, device dtype, and ordered component sum produce the continuum slabs
consumed by the standard synthesis pipeline?

**Equation.**

\[
(\kappa_\lambda^{\rm abs},\kappa_\lambda^{\rm sca})
=\operatorname{continuum}
(\lambda_{\rm requested},\mathrm{atmosphere},\mathrm{tables}),
\]

followed by the derived downstream extinction scale

\[
\texttt{continuum_opacity}
=\texttt{continuum_absorption}
+\texttt{continuum_scattering}.
\]

This sum is not a third independently computed physical slab.

**Exact production names, when earned.** Reveal `ContinuumTables`,
`build_pops`, `build_edge_sample_frequencies`, and `continuum`.

**Inputs, units, and shapes.**

The standard route receives the exact trimmed 18-field view:

- `temperature`, `mass_density`, `electron_density`, each `(D,)`;
- normalized H stages `(D,2)` and actual neutral H/He named vectors;
- normalized C stages `(D,2)` and named neutral Mg/Al/Si/Fe vectors;
- `partition_normalized_populations` and `ion_stage_populations`,
  each `(D,6,139)`;
- four edge-geometry arrays with shapes `(341,)`, `(341,)`, `(340,)`,
  and `(340,)`.

The requested grid is `(W,)` nm. Outputs are `(D,W)` Torch tensors in explicit
CPU/CUDA float64 or MPS float32, cm\(^{2}\) g\(^{-1}\). The standard product
returns no source array.

**Smallest honest implementation.** Validate the trimmed view, call
`build_pops`, load `ContinuumTables` with explicit resolved device and dtype,
and call `continuum` on a compact requested wavelength window. Retain the
ordered sample-column component arrays before edge interpolation.

**Analytic or limiting check.** Every wavelength maps to one clipped interval;
only used intervals are evaluated; absorption and scattering remain separate
and finite/nonnegative; the derived `continuum_opacity` equals their exact
sum. The trimmed route omits actual
`hydrogen_ionized_population`, so `build_pops` uses the normalized H-stage
fallback. It also omits and does not consume stored
`molecular_hydrogen_population`.

**Intended plot.** None at this boundary; H11 already visualizes the
interpolation.

**Exact downstream production parity gate.** Across all four regimes, compare
`build_pops` outputs, named sample-column components, ordered partial sums,
interpolation reconstruction, and final `(D,W)` absorption and scattering
with the pinned standard-pipeline oracle. A call trace must prove
`coulomb_table_energy_first=False` and no `FrequencyInvariants`. Available
backend comparisons use separately declared dtype tolerances.

### B3 — Sampled diagnostic boundary

**Physical question.** How can we inspect exact continuum components at a
small caller-chosen frequency set without claiming that this is the standard
synthesis algorithm?

**Equation.**

\[
(\kappa_\nu^{\rm abs},\kappa_\nu^{\rm sca},B_\nu)
=\operatorname{compute\_sampled\_continuum}
(\mathrm{tables},\nu,\mathrm{pops},\mathrm{None}).
\]

**Exact production names, when earned.** Reveal `build_pops` or, for an
isolated EOS state, `pops_from_population_state`, followed by
`compute_sampled_continuum`.

**Inputs, units, and shapes.**

- exact `pops` device dictionary constructed from a declared state;
- caller frequency `(F,)`, Hz;
- `ContinuumTables` on an explicit device and dtype;
- no `FrequencyInvariants`;
- absorption, scattering, and Planck source `(D,F)` in the work dtype;
  opacities are cm\(^{2}\) g\(^{-1}\), source is
  erg s\(^{-1}\) cm\(^{-2}\) sr\(^{-1}\) Hz\(^{-1}\).

**Smallest honest implementation.** Select only threshold, ownership, and
component-budget probe frequencies, construct `pops`, and call the exact
function without invariants. Label the result “sampled diagnostic” in the
output itself.

**Analytic or limiting check.** Its returned source matches a direct stable
\(B_\nu(T)\) calculation. Named component probes satisfy H03--H07. The route
uses `coulomb_table_energy_first=True`, which is not the standard product
layout.

**Intended plot.** None. Its dense values may supply H11's contextual direct
curve, but that plot remains an interpolation lesson.

**Exact downstream production parity gate.** Compare named components,
ordered sums, absorption, scattering, and source with the pinned sampled
diagnostic oracle on the declared probe grid. Do not compare the whole slab
with B2 as though differing process sets and layouts were aliases.

### B4 — Precomputed sampled extension boundary

**Physical question.** What changes when frequency-only lookups and the fuller
sampled process set are materialized once for a fixed caller-frequency grid?

**Equation.**

\[
I_\nu=\operatorname{build\_frequency\_invariants}
(\mathrm{tables},\nu,
\texttt{coulomb\_table\_energy\_first=True}),
\]

\[
(\kappa_\nu^{\rm abs},\kappa_\nu^{\rm sca},B_\nu)
=\operatorname{compute\_sampled\_continuum}
(\mathrm{tables},\nu,\mathrm{pops},I_\nu).
\]

**Exact production names, when earned.** Reveal `FrequencyInvariants`,
`build_frequency_invariants`, and its `tensor` cache only after the reader
understands which values depend on frequency alone.

**Inputs, units, and shapes.**

- the same fixed caller frequency `(F,)`, exact `pops`, device, dtype, and
  `ContinuumTables` as B3;
- a `FrequencyInvariants` object whose arrays all have `F` as their frequency
  axis and whose stored `coulomb_table_energy_first` is exactly `True`;
- outputs `(D,F)` with the same units as B3;
- a separately declared wavelength/frequency support window in which finite
  whole-slab behavior has been established.

**Smallest honest implementation.** Build invariants once, inspect a compact
shape/layout table, call the sampled continuum with that exact object twice,
and show reuse of cached device tensor views. Do not insert the object into
the standard synthesis pipeline.

**Analytic or limiting check.** Changing the sampled frequency grid invalidates
the invariant identity. A layout mismatch fails loudly. Repeated calls reuse
frequency-only state but not star-dependent populations. All outputs in the
declared support window are finite and nonnegative.

**Intended plot.** None.

**Exact downstream production parity gate.** Compare the full
`FrequencyInvariants` payload, named materialized components, final arrays,
and repeated-call identity with the pinned extension oracle. Compare B3 and
B4 only for specifically shared component identities and within the validated
domain; do not claim global equivalence. A standard-pipeline trace must prove
that B2 never receives this object.

## 6. Exact route matrix shown after B1--B4

The reader sees this only after the four boundaries have been executed:

| boundary | state view | sampling policy | Coulomb layout / invariants | exact outputs |
| --- | --- | --- | --- | --- |
| B1 atmosphere product | packed 18-field `ContinuumAtmosphereState` | every point on direct 30,000 grid | atmosphere NumPy/Numba policy | absorption, scattering, absorption-weighted source `(D,30000)` |
| B2 synthesis product | trimmed 18-field view from schema v4 | three samples for each used edge interval, then log-parabolic interpolation | `False`; no `FrequencyInvariants` | absorption, scattering `(D,W)` |
| B3 sampled diagnostic | exact `pops` dictionary | caller frequencies | `True`; no invariants | absorption, scattering, \(B_\nu\) `(D,F)` |
| B4 sampled extension | same star state plus frequency-only object | caller frequencies with materialized grids | `True`; explicit `FrequencyInvariants` | independently checked absorption, scattering, \(B_\nu\) `(D,F)` |

The atmosphere 343/344-point line-reference result remains H10, a subroute of
B1. It is not a fifth row.

## 7. Existing runtime coverage and required gaps

The current `book/chapter05_runtime.py` already supplies trustworthy starting
points:

- H01 through `mass_opacity_from_cross_section` and
  `opacity_scaling_checkpoint`;
- H02 through `stimulated_emission_factor` and
  `stimulated_emission_checkpoint`;
- H03 through `hminus_edge_checkpoint`, including separately retained
  bound-free/free-free arrays;
- H04--H05 through `atmosphere_component_budget` and
  `metal_population_ownership_checkpoint`, including all fourteen named
  atmosphere absorption components, source reconstruction, and independent
  normalized/actual hot-metal limits;
- H06 through `molecular_continuum_checkpoint`, including CH/OH/CIA ownership,
  the 9000 K per-layer gate, the 20000 cm\(^{-1}\) CIA boundary, and
  `synthesis_stored_h2_invariance_checkpoint`;
- H07 through `scattering_checkpoint`, including separately gated Thomson and
  H/He/H2 Rayleigh arrays and exact cap behavior;
- H08 through `atmosphere_grid_checkpoint` and
  `atmosphere_grid_boundary_checkpoint`, including both sides of all four
  temperature boundaries;
- most of H10 through `line_reference_checkpoint`, including all five active
  counts, `(D,344)` dtype, duplicate, and sentinel;
- H09's transparent Python/serial-`njit`/`prange` equality kernel through
  `book/chapter05_teaching.py` and its isolated cold/warm/cached-process/thread
  evidence through `numba_timing_checkpoint`;
- H11's exact basis, node reproduction, positive log reconstruction, stored
  sample identity, and sign invariance through `edge_triplet_checkpoint` plus
  `book/chapter05_teaching.py`, with exact-edge assignment and unused-interval
  trace through `edge_use_trace_checkpoint`;
- compact exact B1 component-grid calls through
  `run_atmosphere_continuum`;
- compact standard B2 calls through `run_synthesis_continuum`;
- explicit B3/B4 local calls through `run_sampled_synthesis_continuum`, with
  route labels, Coulomb layout, invariant shapes, and tensor-view reuse;
- exact 27→18 synthesis projection and separate atmosphere-adapter evidence
  through `state_projection_checkpoint`;
- input-only regime loading and shape inspection.

Those names should be extended only when a missing gate genuinely needs an
executable checkpoint. The implementation pass must not create a parallel
continuum API.

Before Chapter 5 prose is accepted, the following gaps remain mandatory:

1. the dense molecular/global-gate and IFOP seam matrix in focused tests
   without printing it as one compound notebook cell;
2. B1's independently accepted full `(D,30000)` integration oracle and compact
   reader subset;
3. B2's retained `build_pops`, independently split sample-column components,
   and accepted reconstruction evidence;
4. independently accepted B3/B4 goldens inside the declared extension support
   window;
5. fail-closed elementwise residual, output-schema, worker/contract identity,
   and physical-payload policies in the oracle publisher;
6. four-regime comparison products opened only after each reader-built result
   exists.

## 8. Chapter-boundary guardrails

### Reused without re-teaching

- From Chapter 3: actual versus partition-normalized populations. Chapter 5
  gives one local ownership reminder and perturbation, not a Saha derivation.
- From Chapter 4: CH/OH/H2 chemical availability. Chapter 5 consumes declared
  fields and teaches only the continuum-local H2 policy.
- From Chapter 2: generic array, manifest, Numba, and Torch vocabulary.
  Chapter 5 explains only why this real frequency-column workload is safe and
  how its timing is measured.

### Deferred without previewing implementation

- oscillator strengths, bound-level line strength, Doppler width, damping,
  and line profiles;
- the later line-catalog selection inequality that consumes H10;
- formal transfer with scattering;
- Rosseland-mean physics and atmosphere iteration.

The optional IFOP-19 surrogate is branch-tested for source coverage but is
not used to introduce Rosseland physics.

### Closing dependency

After B1--B4, the reader owns checked smooth absorption and scattering slabs,
their exact state views, and their exact sampling policies. These objects still
cannot create the narrow forest in the opening spectrum. That single missing
physical dependency—not detached practice or an API preview—opens Chapter 6: one
bound-bound transition with a strength, width, and normalized profile.
