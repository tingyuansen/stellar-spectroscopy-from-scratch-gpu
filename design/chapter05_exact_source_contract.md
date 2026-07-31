# Chapter 5 exact construction contract

Status: frozen pre-implementation contract  
Chapter: 5, **Continuous Opacity and Scattering**  
Pinned Payne Zero commit: `9c44001feae40b85146630499e6f8a5fed42e5af`

This contract reconciles the independent atmosphere and synthesis source
audits. It is not reader-facing prose. It prevents the chapter from blending
four real execution lanes into one invented “continuum implementation.”

## 1. Reader destination

Starting from the two handoffs closed in Chapter 4—the packed
`ContinuumAtmosphereState` adapter and the separate 27-field schema-v4
synthesis mapping—the reader must build and verify:

1. microscopic absorption and scattering contributions in
   cm\(^2\) g\(^{-1}\);
2. the atmosphere continuum on its direct, effective-temperature-dependent
   30,000-point CPU grid;
3. the atmosphere `(D,344)` float32 line-selection threshold;
4. the standard synthesis continuum on a requested wavelength grid through
   used edge triplets and logarithmic parabolic interpolation;
5. named component budgets whose ordered sums reproduce the pinned lanes.

No line profile, formal transfer solve, Rosseland mean, or atmosphere
temperature iteration belongs here.

## 2. Four lanes that must remain distinct

| Lane | Exact entry | Grid and work policy | Output | Chapter role |
|---|---|---|---|---|
| atmosphere product | `compute_continuum_opacity_columns` and caller in `runner.prepare_opacity_state` | direct 30,000-point NumPy/Numba CPU float64 | absorption, scattering, absorption-weighted source `(D,30000)` | atmosphere parity destination |
| synthesis product | `continuum` from `SynthesisPipeline.run` | three samples for each used edge interval; `coulomb_table_energy_first=False`; no `FrequencyInvariants` | absorption and scattering `(D,W)` in device work dtype | synthesis parity destination |
| sampled diagnostic | `compute_sampled_continuum(..., frequency_invariants=None)` | caller frequencies; `coulomb_table_energy_first=True` | absorption, scattering, Planck source `(D,F)` | compact component experiments and goldens |
| sampled extension | `build_frequency_invariants` plus `compute_sampled_continuum` | precomputed full frequency grids; alternate light-element/metal helpers | same sampled outputs | separately labeled implemented extension |

The last two lanes are not performance-only aliases of the synthesis product.
They differ in table layout, process set, constants, and observed numerical
behavior. The notebook may compare them but must never report their union as
the production synthesis algorithm.

## 3. Shared physical spine

The causal derivation is shared once:

\[
\kappa_\nu = \frac{n_{\rm absorber}\sigma_\nu}{\rho},
\qquad
s_\nu(T)=1-\exp\left(-\frac{h\nu}{kT}\right).
\]

The first identity converts cross section to mass opacity. The second converts
gross absorption to net LTE absorption when that factor is not already owned
by a fitted coefficient. Partition-normalized populations feed explicit
bound-level sums; actual populations feed free-particle, charge-square, and
scattering terms.

Absorption and scattering remain separate. Scattering redirects photons and
never enters the atmosphere thermal-source numerator. Chapter 9 owns their
coupling in transfer.

## 4. Exact process ownership

### 4.1 Processes in both product lanes

- H-minus bound-free and free-free;
- H I bound-free, high-level tail, and H II free-free;
- H2-plus absorption;
- He-minus absorption;
- He I and He II bound-free/free-free families;
- neutral-metal continua, with lane-specific compact/full implementations;
- hot-metal bound-free and charge-square free-free;
- Si II continuum, with different atmosphere and synthesis tables;
- Thomson, H I Rayleigh, He I Rayleigh, and H2 Rayleigh scattering.

“Present in both” does not imply cross-lane array equality. Each lane retains
its constants, state adapter, floors, interpolation, and sum order.

### 4.2 Atmosphere-product-only processes

- CH photodissociation;
- OH photodissociation;
- H2-H2 and H2-He collision-induced absorption.

These processes are taught physically once, executed through the atmosphere
lane, and proved absent from the synthesis product call trace. They must not be
added to synthesis in the name of completeness.

### 4.3 Sampled-extension-only processes

The explicit N I, O I, Mg II, and Ca II light-element grids and the fully
materialized neutral-metal helpers belong to the precomputed sampled
extension. They do not run in `SynthesisPipeline.run`.

## 5. H2 ownership decision

Four H2 representations coexist:

1. Chapter 4's atmosphere molecular-equilibrium population;
2. schema-v4 `molecular_hydrogen_population`;
3. the atmosphere-continuum local reconstruction using the 200-point H2
   partition table;
4. the synthesis-continuum local analytic reconstruction.

The two continuum product lanes read neither item 1 nor item 2 for H2
Rayleigh/CIA. The schema field remains legitimate state for later consumers,
but it is not a continuum input.

Atmosphere CH/OH availability follows `enable_molecules`; the local H2
reconstruction does not. The adapter names `ch_population` and `oh_population`
are exact aliases for partition-normalized packed slots 845 and 847. The
cross-section helpers carry the matching partition factor, so these fields
must not be described as generic actual molecular densities. A
molecules-disabled state can therefore have zero CH/OH continuum and nonzero
H2 CIA/Rayleigh.

## 6. Resolved source ambiguities

These decisions are binding.

1. **H2-CIA temperature weights.** Preserve the pinned, reversed-looking
   lower/upper weight order. Capture exact lower nodes, upper nodes, and one
   midpoint. Label it a parity-pinned convention; do not silently “fix” it.
2. **IFOP 4/13 coupling.** H2 Rayleigh is operational only when IFOP 4 has
   constructed the H I ground population and IFOP 13 is enabled. Teach and
   test the combined policy.
3. **No repair clamp.** Reader-facing validators reject nonpositive,
   nonfinite, or wrongly shaped physical inputs before exact kernels run.
   Valid fixtures must produce finite nonnegative components without a final
   mutation clamp.
4. **Loader-required versus consumed fields.** Gavrila and Peach fields remain
   in the atmosphere archive for loader identity but are labeled
   `required_for_loader`, not `consumed_by_atmosphere_process`.
5. **Inactive level table.** `continuum_level_tables.npz` is excluded from the
   active teaching/runtime bundle unless a later trace establishes a consumer.
6. **H-minus stored unit.** The exact field name remains
   `hminus_boundfree_cross_section_cm2`, but its raw numbers are declared in
   units of \(10^{-18}\) cm\(^2\); both consumers multiply by `1e-18`.
7. **Rosseland surrogate.** IFOP 19 is source-covered and tested separately,
   but excluded from the standard physical continuum. Its bolometric-like
   source is not blended into the chapter's \(B_\nu\) source lesson.
8. **Explicit flags.** Atmosphere calls always pass the pinned runner vector;
   `opacity_flags=None` is not treated as the runner default.
9. **Column-global molecular gate.** Entering the atmosphere molecular
   composite depends on `min(temperature) < 9000 K`; CH/OH then gate each
   layer at 9000 K while H2 CIA uses its separate 20,000 K rule.
10. **Synthesis standard route.** Product parity targets `continuum(...)`
    with the trimmed pipeline mapping, no `FrequencyInvariants`, and Coulomb
    layout `False`.
11. **Supported extension window.** Whole-slab extension comparisons are
    limited to wavelengths for which finite behavior is established.
    Component tests may probe wider physical domains without claiming global
    equivalence.
12. **H II mapping view.** Product goldens use the trimmed pipeline view and
    its normalized-stage fallback. A richer direct mapping is a separately
    labeled comparison.
13. **Signed edges.** Preserve and hash the signed stored field; the current
    sampler uses its magnitude. A sign-invariance test pins this fact.
14. **Dtype defaults.** Product code passes the resolved device/dtype
    explicitly. The standalone table loader's omitted CPU dtype of float32 is
    demonstrated once and never confused with the pipeline's float64 default.

## 7. Data and source bundle

Exact staged source files:

- `src/payne_zero_atmosphere/continuum_opacity.py`, SHA-256
  `1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81`;
- `src/payne_zero_synthesis/continuum.py`, SHA-256
  `ab0d4eb771ee04101f6936253f633ed60d845e2816854a06b1b059e8b91dce1b`.

Exact active static files:

- atmosphere `continuum_opacity_tables.npz`, SHA-256
  `6fd4c556418870c28d3fcc9a050252af58ac4cc433cae979477355c8c7d593e3`;
- atmosphere `karzas_latter_tables.npz`, SHA-256
  `23805dc17c47af45b8ae63b2e278e1fb6c584a01c87d1eb3c31306e4555e6d15`;
- atmosphere `molecular_equilibrium_tables.npz`, already staged;
- synthesis `continuum_tables.npz`, SHA-256
  `406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d`;
- synthesis `continuum_edge_grid.npz`, already staged.

Every NPZ member has a shape, dtype, unit/convention, byte hash, source path,
source hash, and consumer role in `data/MANIFEST.json`.

## 7.1 Exact reads contracts

The atmosphere product receives an 18-field `ContinuumAtmosphereState`, not
schema v4:

| field group | exact fields | shape and role |
| --- | --- | --- |
| thermodynamic | `temperature`, `mass_density`, `electron_density`, `gas_pressure`, `microturbulence` | each `(D,)`; K, g cm\(^{-3}\), cm\(^{-3}\), dyn cm\(^{-2}\), cm s\(^{-1}\) |
| H stage | `hydrogen_partition_normalized_ion_stage_populations` | `(D,2)`, cm\(^{-3}\) per partition |
| actual H/He | `hydrogen_neutral_population`, `hydrogen_ionized_population`, `helium_neutral_population`, `helium_singly_ionized_population` | each `(D,)`, cm\(^{-3}\) |
| normalized He | `helium_neutral_partition_normalized_population`, `helium_singly_ionized_partition_normalized_population` | each `(D,)`, cm\(^{-3}\) |
| layer mixture | `elemental_abundances_by_layer` | `(D,99)`, nuclei fraction |
| departure state | `hydrogen_departure_coefficients` | `(D,6)`, dimensionless |
| packed populations | `ion_stage_populations_by_packed_slot`, `partition_normalized_populations_by_packed_slot` | each `(D,1006)`, cm\(^{-3}\) |
| normalized molecules | `ch_population`, `oh_population` | each `(D,)`, exact aliases of packed normalized slots 845/847 |

The standard synthesis product receives the exact 18-field trimmed pipeline
view:

| field group | exact fields | shape and role |
| --- | --- | --- |
| thermodynamic | `temperature`, `mass_density`, `electron_density` | each `(D,)` |
| named H/He | `hydrogen_partition_normalized_ion_stage_populations`, `hydrogen_neutral_population`, `helium_neutral_population`, `helium_singly_ionized_population` | `(D,2)` then three `(D,)` vectors |
| named metals | `carbon_partition_normalized_ion_stage_populations`, `magnesium_neutral_partition_normalized_population`, `aluminum_neutral_partition_normalized_population`, `silicon_neutral_partition_normalized_population`, `iron_neutral_partition_normalized_population` | `(D,2)` then four `(D,)` vectors |
| population cubes | `partition_normalized_populations`, `ion_stage_populations` | each `(D,6,139)` |
| edge geometry | `signed_continuum_edge_frequency_hz`, `continuum_edge_wavelength_nm`, `continuum_edge_midpoint_wavelength_nm`, `continuum_edge_interval_width_squared_over_two_nm2` | `(341,)`, `(341,)`, `(340,)`, `(340,)` |

The trimmed synthesis view deliberately omits actual
`hydrogen_ionized_population` and stored
`molecular_hydrogen_population`. `build_pops` therefore uses its exact
normalized-H-stage fallback for H II, while H2 Rayleigh is reconstructed
locally.

This 18-field mapping is an exact consumer projection from the upstream
27-field schema-v4 synthesis atmosphere. It is not a replacement public
schema, and it is not the packed 18-field `ContinuumAtmosphereState`. The
chapter must execute this projection once and must never reconstruct either
18-field consumer view from the other.

## 8. Causal chapter movements

### Act I — A cross section becomes absorption

1. Return briefly to the already-built Chapter 1 spectrum and ask what can
   keep the gas nontransparent between the narrow dips. Do not introduce
   bound-bound terminology, preload a Chapter 5 result, or open a comparison
   golden.
2. Derive `n * sigma / rho` and check units.
3. Distinguish bound-free and free-free with one threshold schematic.
4. Derive stimulated emission once.
5. Just before a physical table is first consumed, verify the manifest-bound
   H-minus inputs.
6. Explain the local H-minus population factor from temperature, normalized
   neutral H, electron density, and unit LTE departure factors; then build
   H-minus bound-free/free-free.
7. Build one H I level, then the explicit ladder, tail, and H II free-free.
8. Add H2-plus, He-minus, He I, and He II.
9. Close with an ordered H/He budget at a solar and hot depth.

### Act II — Named processes build the physical budget

1. Use the missing near-UV budget to motivate neutral metals.
2. Add compact neutral metals, hot metals, and Si II in exact product order.
3. Perturb normalized and actual population views independently.
4. Introduce CH, OH, and H2 CIA in the cool atmosphere and state their product
   boundary immediately.
5. Close with regime-specific named absorption budgets.

### Act III — Scattering redirects; two exact grids place the answer

1. Derive Thomson scaling.
2. Compare H I, He I, and H2 Rayleigh color trends.
3. Keep absorption and scattering as separate slabs.
4. Build the five atmosphere 30,000-point grid regimes.
5. Build the 343/344-point atmosphere line-selection threshold.
6. Draw the synthesis one-sided edge triplet.
7. Derive and verify the three parabolic basis functions.
8. Interpolate floored log opacity only across used intervals.
9. Close with the atmosphere line-reference threshold as a subroute of the
   atmosphere product, not as a fifth lane.

### Act IV — Assemble the two products and prove their boundaries

1. Execute the projection from the 27-field schema-v4 synthesis atmosphere to
   its exact 18-field continuum view, and contrast it with the separate
   18-field packed atmosphere adapter.
2. Preserve the exact atmosphere IFOP vector and ordered component sums.
3. Preserve the standard synthesis route with Coulomb layout `False` and no
   `FrequencyInvariants`.
4. Compare named components before final slabs for all four regimes.
5. Close with four-lane parity tables and the derived synthesis
   `continuum_opacity = continuum_absorption + continuum_scattering` handoff.

## 9. Visible-code budget

Target 13–14 visible code cells, with 16 as a hard ceiling. Each cell is no
more than 35 lines and has one conceptual purpose:

1. microscopic-to-mass-opacity units and scaling;
2. stable stimulated-emission limits;
3. just-in-time manifest/table preflight;
4. H-minus local-population ownership, threshold, and bound-free/free-free sum;
5. solar/hot H/He budget;
6. normalized-versus-actual metal-population ownership and named metal sum;
7. atmosphere-only CH/OH/CIA ownership, one temperature seam, and the standard
   synthesis stored-H2 non-use check;
8. Thomson/Rayleigh limits and the separate scattering slab;
9. atmosphere grid thresholds, weights, and line-reference subroute;
10. transparent Python/serial-`njit`/`prange` column parity and labelled timing;
11. synthesis edge assignment, triplet basis, and one reconstruction;
12. 27-field synthesis-to-18-field continuum projection, separate atmosphere
    adapter, and route-specific dtype/device contract;
13. named atmosphere partial sums, full 30,000-point product, and parity;
14. standard synthesis product parity plus a compact identity table for the
    diagnostic and extension lanes.

Long exact kernels live in the progressive package. Visible cells reconstruct
small transparent laws or call one exact checkpoint; no source block is pasted
into Markdown.

## 10. Visual contract

Create four original schematics in the website's restrained dark-blue,
warm-orange, pale-background aesthetic:

- absorber population → microscopic cross section → mass opacity;
- true absorption versus coherent scattering;
- independent frequency-column ownership under `prange`;
- a combined two-grid schematic whose synthesis branch enlarges one edge
  interval and shows the one-sided left/middle/right samples.

Use one-panel professional plots only. Every axis carries quantity and unit;
color has stable process meaning; no panel exists merely to display more data.
Aim for six plots; grid policies and final parity can be clearer as schematics
or compact numerical tables.

## 11. Verification gates

Before chapter prose:

- pinned source and data identity;
- exact table fields, shapes, dtypes, and active/inactive roles;
- physical-domain validation;
- component linearity in population and inverse linearity in mass density;
- actual-versus-normalized population perturbations;
- all thresholds and H2/IFOP/global-gate seams;
- exact atmosphere sampling thresholds and endpoint weights;
- exact synthesis edge side, used-interval count, basis partition, and node
  reproduction;
- atmosphere product, synthesis product, diagnostic, and extension call traces;
- ordered component and final slab oracles;
- one-thread/many-thread and cold/warm Numba checks;
- CPU float64 and available CUDA float64/MPS float32 checks;
- four stellar regimes with active and physically near-zero branches.

Goldens are comparison-only and opened after the local calculation.

## 12. Closure and handoff

The chapter closes with checked smooth absorption and scattering backgrounds,
not a finished spectrum. Its final question is causal:

> If the continuum explains the broad background, what produces the narrow
> forest of wavelength-localized absorption?

Chapter 6 answers by constructing one bound-bound line from its level
population, oscillator strength, Doppler width, damping, and normalized
profile.
