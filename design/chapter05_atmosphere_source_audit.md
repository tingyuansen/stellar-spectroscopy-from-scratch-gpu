# Chapter 5 atmosphere-continuum source audit

Status: read-only implementation contract, not chapter prose  
Oracle repository: `/Users/ysting/payne-zero`  
Pinned commit: `9c44001feae40b85146630499e6f8a5fed42e5af`  
Audit date: 2026-07-30

This document freezes the atmosphere-side continuum calculation that Chapter
5 must teach and reproduce. It is organized by causal dependency rather than
by repository layout. The pinned source and its data remain read-only.

## 1. Executive decisions

Six distinctions are P0 for the chapter.

1. The atmosphere and synthesis lanes share much of the physical vocabulary,
   but they do not share a grid algorithm. The atmosphere evaluates continuum
   opacity directly on a 30,000-point, \(10^{-4}\)-dex wavelength grid. The
   three-sample edge interpolation belongs to synthesis.
2. The atmosphere produces three `float64` arrays with axes `(depth,
   frequency)`: mass absorption, mass scattering, and the absorption-weighted
   thermal source. It does not merge absorption and scattering.
3. CH photodissociation, OH photodissociation, and H2 collision-induced
   absorption are active atmosphere-only terms. They are absent from the
   pinned synthesis `continuum(...)` composite.
4. The atmosphere continuum code does not consume the Chapter 4 H2 output
   directly. It reconstructs an H2 density from neutral hydrogen for H2
   collision-induced absorption and H2 Rayleigh scattering. This bridge has
   its own temperature policy and must be taught as an exact implementation
   boundary, not as a second derivation of molecular equilibrium.
5. The atmosphere CPU path requires Numba. Parallel kernels use `prange` only
   over independent frequencies; ordered state construction and assembly stay
   outside those loops.
6. The pinned composite does not clip its final opacity arrays to be finite or
   nonnegative. The book may validate the physical input domain, but it must
   not add an unreported final clamp and then claim exact parity.

The final Chapter 5 artifact therefore has two explicit layouts:

| Lane | Grid | Output |
|---|---|---|
| Atmosphere | direct 30,000-point sampling grid | `continuum_absorption`, `continuum_scattering`, `continuum_source`, each `(D, 30000)` float64 on CPU |
| Synthesis | requested `(W,)` wavelength grid through used edge triplets | `continuum_absorption`, `continuum_scattering`, each `(D, W)` in device work dtype |

The atmosphere-only audit below closes the first row. The synthesis row remains
governed by `design/part3_part4_synthesis_brief.md`.

## 2. Pinned evidence and source anchors

### 2.1 Source identities

| Source | SHA-256 | Git blob | Active responsibility |
|---|---|---|---|
| `payne_zero_atmosphere/continuum_opacity.py` | `1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81` | `6c805f73644c10a79a37f210849d6bda9649d31e` | State adapter, tables, every atmosphere continuum process, scattering, sampling grid, line-selection continuum |
| `payne_zero_atmosphere/runner.py` | `05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7` | `74cf63548f0b93eeff253174af50a43e460021c1` | `prepare_opacity_state(...)` caller and exact output lifecycle |
| `payne_zero_atmosphere/constants.py` | `ac1f1fbd345dc816eb3e70a8f97ebebc7a4c744fd2759b32ec19f8c88d987036` | `cf537e66ae3a414be715aae6d80dceff1fea3560` | Exact and reference constant tiers |
| `payne_zero_atmosphere/population_layout.py` | `36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0` | `f8af79dc9da9d4cd5693857bcc6166b448b4da5c` | Packed atomic and molecular population slots |
| `payne_zero_atmosphere/runtime_state.py` | `fae240ec00f6f89d7c2a7ef721ce6e6539be234e523291fd6e8a096d731430e8` | `ba735961fec83610ce947ff76f2de54d2f89fa37` | Runtime densities and packed population arrays |
| `payne_zero_atmosphere/atmosphere_io.py` | `95c4d2cab230f6925e9404639ecb05b25af8c0c85755ac1ca70d760156a8683e` | `bb991c1b0345dd677c65606ede69bf9f5185f608` | `ModelAtmosphere` depth columns and metadata |
| `payne_zero_atmosphere/run_setup.py` | `de7cf08b936585dbcfa2e572c026fafa3f10282a99c27b834b62db0f3f2888c9` | `48864e3d1f35657391d51b50e212ec936f40f18b` | 20 external-format opacity flags |
| `payne_zero_atmosphere/config.py` | `51e19846fb81c832ae57334faf3da2c1e4fc2ef9edf6e08467ef7296e4640b45` | `b8aaba69dc219a501da57fed51136a2769fb93c1` | Default flag vector |
| `payne_zero_atmosphere/data_files.py` | `bf89c32977fc2db0454cf597718d99b3f3d15487529ecddbacf717ad6dc245c2` | `d257725dcf9ed24b19c45337505ae420924d0cf3` | `PAYNE_ZERO_DATA_ROOT` and strict required-key loader |
| `payne_zero_atmosphere/_numba_cache.py` | `b8988812ea92fd5db1e7f092d06ce685e13fbda3b0b7910eba937eb7a4ddeb82` | `f11084adc9f5b38543123e649622214ae9137a1f` | Persistent Numba cache selection |
| `payne_zero_atmosphere/microturbulence.py` | `3692062f1d6877e745ed84bba4fc2fdf04c60a7c52bc27856fc696416a0283cb` | `21a76c03a54dad24faf72edae4d5d31086048bcb` | Exact piecewise-quadratic H-minus bound-free remap |

The reconciliation reference is
`payne_zero_synthesis/continuum.py`, SHA-256
`ab0d4eb771ee04101f6936253f633ed60d845e2816854a06b1b059e8b91dce1b`,
Git blob `e197bb648a1c890c30bca65aaf57dba95231e7e3`.

### 2.2 Semantic anchors in `continuum_opacity.py`

| Lines | Contract |
|---:|---|
| 757–989 | Table/state dataclasses, exact required keys and shapes |
| 991–1084 | Table validation and cached loaders |
| 1087–1183 | `ContinuumAtmosphereState` construction and packed-slot mapping |
| 1186–1294 | 343/344-point line-selection continuum grid and threshold |
| 1297–1399 | Atmosphere-local H2 bridge and recomputed H I ground population |
| 1402–1488 | Linear interpolation/extrapolation and exact Planck/stimulated-emission grid |
| 1513–2519 | Numba kernels and their parallel axes |
| 2525–2865 | Karzas-Latter, He I transition, and Coulomb-Gaunt helpers |
| 2868–3520 | H I, He I, H-minus, H2-plus, He-minus, and He II |
| 3523–4670 | C I, Mg I, Si I, and generated Si II table |
| 4686–4953 | N I/O I/C II/Mg II/Si II/Ca II lukewarm-metal composite |
| 4955–5257 | CH, OH, and H2 collision-induced absorption |
| 5279–5707 | Hot-metal, Al I, and Fe I continua |
| 5709–5832 | Optional Rosseland-table surrogate |
| 5834–6054 | Complete and light-element absorption/source assembly |
| 6056–6156 | Electron and H I/He I/H2 scattering |
| 6158–6197 | 30,000-point atmosphere opacity-sampling grid |

The production caller is `runner.py:650–799`. It builds one
`ContinuumAtmosphereState`, computes the 30,000-point continuum, recomputes a
smaller reference continuum for line selection, and stores both in
`OpacityState`.

## 3. Static data contract

### 3.1 Active archives

| Archive | SHA-256 | Active role |
|---|---|---|
| `atmosphere_tables/continuum_opacity_tables.npz` | `6fd4c556418870c28d3fcc9a050252af58ac4cc433cae979477355c8c7d593e3` | H-minus, Coulomb free-free, hot metals, CH/OH, H2 collisions, H I partition helper |
| `atmosphere_tables/karzas_latter_tables.npz` | `23805dc17c47af45b8ae63b2e278e1fb6c584a01c87d1eb3c31306e4555e6d15` | Hydrogenic bound-free cross sections |
| `atmosphere_tables/molecular_equilibrium_tables.npz` | `1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe` | Atmosphere-local H2 partition function |

`continuum_level_tables.npz`, SHA-256
`35a6839be4ff3dd824206c7a6b851b987132313374ede7ea5441f9d0bd69888f`,
has an exported loader but is never called by the pinned atmosphere continuum
path. It is not an active Chapter 5 runtime dependency.

### 3.2 `continuum_opacity_tables.npz`

Every array is float64. The loader requires every key below even when the
atmosphere process code does not read that field.

| Fields | Shape | Atmosphere use |
|---|---:|---|
| `coulomb_freefree_charge_log_offset` | `(6,)` | Active, charges 1–6 interpolation offset |
| `coulomb_freefree_gaunt_table` | `(12,11)` | Active Coulomb free-free table |
| `hminus_boundfree_wavelength_nm`, `hminus_boundfree_cross_section_cm2` | `(85,)` each | Active H-minus bound-free remap; despite the second key's suffix, its stored numbers are in \(10^{-18}\ {\rm cm^2}\) units and the routine multiplies by `1e-18` |
| `hminus_freefree_inverse_wavelength_grid` | `(22,)` | Active H-minus free-free wavelength coordinate |
| `hminus_freefree_theta_grid` | `(11,)` | Active \(\theta=5040/T\) coordinate |
| `hminus_freefree_short_wavelength_table`, `hminus_freefree_long_wavelength_table` | `(11,11)` each | Active H-minus free-free values |
| `hot_metal_boundfree_transition_table` | `(60,7)` | Active threshold, cross section, power, multiplier, excitation, population-slot records |
| `ch_partition_table`, `oh_partition_table` | `(41,)` each | Active molecular partition interpolation |
| `ch_cross_section_table` | `(106,15)` | Active CH cross sections |
| `oh_cross_section_table` | `(130,15)` | Active OH cross sections |
| `hydrogen_molecule_h2_collision_table`, `hydrogen_molecule_he_collision_table` | `(81,7)` each | Active H2-H2 and H2-He collision-induced absorption |
| `hydrogen_neutral_level_energy_cm`, `hydrogen_neutral_level_statistical_weight` | `(6,)` each | Active recomputation of H I partition-normalized ground population |
| six `hydrogen_rayleigh_gavrila_*` arrays | `(74,)`, `(27,)`, `(24,)`, `(22,)`, `(64,)`, `(64,)` | Loader-required but not read by atmosphere; atmosphere H I Rayleigh uses a literal analytic fit |
| four `silicon_singly_ionized_peach_*` arrays | `(14,6)`, `(7,)`, `(9,)`, `(6,)` | Loader-required but not read by atmosphere; atmosphere builds its own cached `(200,51)` Si II table |

The final two groups are active in the synthesis implementation, not in the
atmosphere implementation. A copied atmosphere archive must retain them for
loader parity, but chapter prose must not attribute their values to an
atmosphere calculation that never reads them.

### 3.3 Other archives

`karzas_latter_tables.npz` contains:

| Field | Shape | Dtype |
|---|---:|---|
| `karzas_latter_log10_frequency_hz` | `(29,15)` | float64 |
| `karzas_latter_total_log10_cross_section_cm2` | `(29,15)` | float64 |
| `karzas_latter_angular_log10_cross_section_cm2` | `(6,6,29)` | float64 |
| `karzas_latter_high_level_energy_offset_rydberg` | `(29,)` | float64 |

The loader validates all four shapes.

`molecular_equilibrium_tables.npz` contains
`h2_partition_function (200,) float64`, which is loaded and shape-checked.
Its `atomic_mass_amu (99,) float64` field is present but not requested by this
loader.

### 3.4 Table-axis semantics used by the kernels

| Array | Axis order |
|---|---|
| `karzas_latter_log10_frequency_hz`, `karzas_latter_total_log10_cross_section_cm2` | `(frequency_node, principal_quantum_number_1_through_15)` |
| `karzas_latter_angular_log10_cross_section_cm2` | `(orbital_angular_momentum_0_through_5, principal_quantum_number_1_through_6, frequency_node)` |
| H-minus short/long free-free tables | `(wavelength_index_within_half, theta_index)`; the routine concatenates the halves into 22 wavelength rows and transposes to `(theta_index, wavelength_index)` |
| `hot_metal_boundfree_transition_table` | `(transition, field)` with fields `threshold_frequency_hz`, `threshold_cross_section_cm2`, `alpha`, `power`, `multiplier`, `excitation_energy_ev`, `population_index_1based` |
| `ch_cross_section_table`, `oh_cross_section_table` | `(photon_energy_index, temperature_index)` and stored base-10 logarithmic cross section |
| H2 collision tables | `(wavenumber_index_in_250_cm^-1_steps, temperature_index_in_1000_K_steps)` and stored base-10 logarithmic binary coefficient |
| `h2_partition_function` | `(temperature_index_in_100_K_steps,)` |

### 3.5 Pinned constant tiers

The atmosphere package intentionally mixes exact CODATA literals with rounded
reference literals inherited by particular fits. They are not interchangeable.

| Exact name | Value | Use |
|---|---:|---|
| `LIGHT_SPEED_CM_PER_S_EXACT` | `2.99792458e10` cm s^-1 | Planck, thresholds, wavenumbers, Karzas and helium |
| `LIGHT_SPEED_NM_PER_S` | `2.99792458e17` nm s^-1 | wavelength/frequency grids |
| `LIGHT_SPEED_ANGSTROM_PER_S` | `2.99792458e18` Å s^-1 | Rayleigh fits |
| `PLANCK_ERG_SECOND_EXACT` | `6.62607015e-27` erg s | exact Planck and Boltzmann factors |
| `BOLTZMANN_ERG_PER_K_EXACT` | `1.380649e-16` erg K^-1 | exact Planck and Boltzmann factors |
| `LIGHT_SPEED_CM_PER_S_REFERENCE` | `2.997925e10` cm s^-1 | parity-pinned H2 and table fits |
| `PLANCK_ERG_SECOND_REFERENCE` | `6.6256e-27` erg s | parity-pinned H2, Si II, and line threshold |
| `BOLTZMANN_ERG_PER_K_REFERENCE` | `1.38054e-16` erg K^-1 | parity-pinned H2, Si II, and line threshold |
| `BOLTZMANN_EV_PER_K_REFERENCE` | `8.6171e-5` eV K^-1 | thermal energies in atomic fits |
| `ATOMIC_MASS_GRAM_REFERENCE` | `1.660e-24` g | H2 translational factor |
| `WAVENUMBER_PER_EV_REFERENCE` | `8065.479` cm^-1 eV^-1 | atmosphere continuum energy conversion |
| `REFERENCE_NATURAL_LOG_10` | `2.30258509299405` | base-10 table conversion |

The atmosphere value `WAVENUMBER_PER_EV_REFERENCE` must not be replaced by
the synthesis line-catalog conversion. Constant-tier substitution receives an
independent parity gate.

## 4. Exact state, axes, units, and population roles

Let `D` be the number of layers, ordered outermost to innermost, and let `F`
be the number of queried frequencies.

### 4.1 Inputs

| Exact field | Shape | Unit / role |
|---|---:|---|
| `temperature` | `(D,)` | K |
| `mass_density` | `(D,)` | g cm^-3 |
| `electron_density` | `(D,)` | cm^-3, actual |
| `gas_pressure` | `(D,)` | dyn cm^-2 |
| `hydrogen_partition_normalized_ion_stage_populations` | `(D,2)` | cm^-3 per partition function |
| `hydrogen_neutral_population`, `hydrogen_ionized_population` | `(D,)` each | cm^-3, actual |
| `helium_neutral_population`, `helium_singly_ionized_population` | `(D,)` each | cm^-3, actual |
| `helium_neutral_partition_normalized_population`, `helium_singly_ionized_partition_normalized_population` | `(D,)` each | cm^-3 per partition function |
| `elemental_abundances_by_layer` | `(D,99)` | number fraction; carried but not read by current continuum functions |
| `hydrogen_departure_coefficients` | `(D,N)`, normally `N>=6` | dimensionless; defaults to ones if absent |
| `microturbulence` | `(D,)` | cm s^-1; carried but not read by continuum |
| `ion_stage_populations_by_packed_slot` | `(D,P)` | actual cm^-3 |
| `partition_normalized_populations_by_packed_slot` | `(D,P)` | cm^-3 per partition function |
| `ch_population` | `(D,)` | packed normalized slot 845 zero-based, code 106 / CH |
| `oh_population` | `(D,)` | packed normalized slot 847 zero-based, code 108 / OH |
| `frequency_hz` | `(F,)` | Hz |

`build_continuum_atmosphere_state(...)` requires matching depth axes, at
least two actual-population columns, and packed normalized slot 847. It maps
H I/H II from slots 0/1, He I/He II from 2/3, and He III later from actual
slot 4.

### 4.2 Outputs

| Exact output | Shape | Unit / meaning |
|---|---:|---|
| `absorption` | `(D,F)` | cm2 g^-1, true absorption |
| `scattering` | `(D,F)` | cm2 g^-1, coherent continuum redirection coefficient |
| `source` | `(D,F)` | \(B_\nu\) units: erg s^-1 cm^-2 sr^-1 Hz^-1 on the standard IFOP-19-off path |

Every process starts from a microscopic cross section or fit and reaches mass
opacity through the pattern

\[
\kappa_\nu = \frac{n_{\rm absorber}\,\sigma_\nu}{\rho}.
\]

Partition-normalized populations weight bound-level Boltzmann factors.
Actual ion populations weight free-free charge densities and electron
scattering. Confusing these two views can preserve plausible magnitudes while
breaking the exact process budget.

The source numerator is

\[
N_\nu = \sum_j \kappa_{\nu,j}^{\rm abs} S_{\nu,j}.
\]

H I and H-minus may have non-LTE departure-coefficient sources. Every other
active continuum absorption term uses \(B_\nu\). Scattering never enters this
thermal numerator. Where total absorption is zero, `source` is set to
`planck_nu`.

## 5. Exact opacity-flag ownership

The default external vector is:

```text
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]
```

Flags are shown below with their external one-based `IFOP` number. The
continuum functions activate a branch only when its value is exactly `1`.

| IFOP | Zero-based index | Meaning in the pinned runner |
|---:|---:|---|
| 1 | 0 | H I bound-free/high-level tail plus H II free-free |
| 2 | 1 | H2-plus absorption |
| 3 | 2 | H-minus bound-free and free-free |
| 4 | 3 | H I Rayleigh; also supplies the reconstructed ground population required by IFOP 13 |
| 5 | 4 | He I continuum |
| 6 | 5 | He II continuum |
| 7 | 6 | He-minus absorption |
| 8 | 7 | He I Rayleigh |
| 9 | 8 | One inseparable group: CH, OH, H2 CIA, C I, Mg I, Al I, Si I, Fe I |
| 10 | 9 | Lukewarm N I/O I/C II/Mg II/Si II/Ca II group |
| 11 | 10 | Hot-metal bound-free and ionic free-free group |
| 12 | 11 | Thomson electron scattering |
| 13 | 12 | H2 Rayleigh, but only if IFOP 4 also constructed the H I ground population |
| 14 | 13 | HLINOP branch; rejected by `run_atmosphere_model` |
| 15 | 14 | Selected-line generation/accumulation, not continuum |
| 16 | 15 | Unused by the pinned runner |
| 17 | 16 | Detailed transition-line accumulation, not continuum |
| 18 | 17 | Unused by the pinned runner |
| 19 | 18 | Optional Rosseland-table surrogate continuum if a table is passed |
| 20 | 19 | Unused by the pinned runner |

`opacity_flags=None` inside the continuum module means twenty ones, not the
runner default. Chapter fixtures and oracles must therefore pass the explicit
runner vector.

## 6. Complete process, routine, population, and data matrix

All listed absorption and scattering outputs use axes `(D,F)`.

| Process | Exact routine | Population role | Cross-section / fit | Active domain and policy | Source |
|---|---|---|---|---|---|
| H I bound-free | `compute_hydrogen_opacity_columns` | normalized H I, explicit shells 1–8; departures for shells 1–6 | Karzas-Latter | Thresholded per shell; shells 7–8 enter the LTE base, 1–6 enter departure-corrected terms | non-LTE-capable |
| H I high-level tail | same | normalized H I | frequency^-3 extension | switches extension treatment below `4.05933e13 Hz` | LTE base |
| H II free-free | same | actual H II times actual electrons | charge-1 Coulomb Gaunt, frequency^-3 | all positive frequencies | LTE base |
| H-minus bound-free | `compute_hminus_opacity_columns` | normalized H I, electrons, H I and H-minus departures | 85-point table in \(10^{-18}\ {\rm cm^2}\), converted by an explicit `1e-18`, and `_piecewise_quadratic_remap` | only `frequency > 1.82365e14 Hz` | non-LTE-capable |
| H-minus free-free | same | normalized H I times electrons | 22 wavelength by 11 theta table | linear interpolation with endpoint extrapolation in log wavelength and theta | Planck |
| H2-plus | `compute_molecular_hydrogen_ion_opacity_columns` | two normalized H stages and H I ground departure | frequency and excitation polynomials | only `frequency <= 3.28805e15 Hz` | Planck |
| He I bound-free | `compute_helium_neutral_opacity_columns` | normalized He I | ten low levels, Karzas high levels, explicit autoionization additions | per-level thresholds | Planck |
| He I tail/free-free | same | normalized He I tail; actual He II times electrons | frequency^-3 extension and charge-1 Gaunt | extension changes below `2.055e14 Hz`; high levels only above `1.25408e16 Hz` | Planck |
| He-minus free-free-like fit | `compute_heminus_opacity_columns` | actual He I times electrons | three frequency polynomials in `T`, `1`, and `1/T` | no explicit calibrated-domain guard | Planck |
| He II bound-free/tail | `compute_helium_ionized_opacity_columns` | normalized He II, shells 1–9 | Karzas charge-squared 4 plus tail | extension changes below `1.31522e14 Hz` | Planck |
| He III free-free | same | actual He III times electrons | charge-2 Gaunt and factor 4 | all positive frequencies | Planck |
| C I | `compute_carbon_neutral_opacity_columns` | packed normalized slot 20 | 25 levels, Karzas, resonances, high-level extension | calculation masked to `frequency <= 3.28805e15 Hz` | Planck |
| Mg I | `compute_magnesium_neutral_opacity_columns` | packed normalized slot 77 | 15 levels, Karzas and analytic edges | same Lyman-frequency mask | Planck |
| Si I | `compute_silicon_neutral_opacity_columns` | packed normalized slot 104 | 33 levels, Karzas and resonances | same Lyman-frequency mask | Planck |
| Al I | `compute_aluminum_neutral_opacity_columns` | packed normalized slot 90 | two analytic edge contributions | same Lyman-frequency mask | Planck |
| Fe I | `compute_iron_neutral_opacity_columns` | packed normalized slot 350 | 48 threshold branches | only wavenumber `>=21000 cm^-1` | Planck |
| N I / O I | `compute_lukewarm_metal_opacity_columns` | normalized N I/O I | Seaton threshold fits | independent frequency columns | Planck |
| C II / Mg II / Ca II | same | corresponding normalized ion stage | Karzas, Kramers tails, Seaton fits | independent frequency columns | Planck |
| Si II | same | normalized Si II | cached generated `(200,51)` log table above `12192.48 cm^-1`, analytic tail below | table bracket clamps in wavenumber and temperature index | Planck |
| Hot-ion free-free | `compute_hot_metal_opacity_columns` | actual stages II–VI of C/N/O/Ne/Mg/Si/S/Fe, summed with charge squared | Gaunt charges 1–5 | frequency chunks of 4096 | Planck |
| Hot-metal bound-free | same | 21 normalized C/N/O/Ne stage slots | 60-record table | a transition is added only where its weighted cross section exceeds the current free-free opacity divided by 100 | Planck |
| CH photodissociation | `compute_molecular_continuum_opacity_columns` → `_ch_molecular_cross_section_grid` | normalized CH packed slot 845 | `(106,15)` log cross sections times `(41,)` partition | photon-energy indices 20–104 and layers `T<9000 K` | Planck |
| OH photodissociation | same → `_oh_molecular_cross_section_grid` | normalized OH packed slot 847 | `(130,15)` log cross sections times `(41,)` partition | positive energy indices below 130 and layers `T<9000 K` | Planck |
| H2-H2 / H2-He CIA | same → `_h2_collision_absorption_grid` | locally reconstructed H2 and actual He I | two `(81,7)` log tables | wavenumber `<=20000 cm^-1`; H2 forced to zero only for `T>20000 K` | Planck |
| Thomson scattering | `compute_continuum_scattering_columns` | actual electrons | `0.6653e-24 cm2` | wavelength-independent at fixed `n_e/rho` | none |
| H I Rayleigh | same | recomputed normalized H I ground times departure | analytic inverse-wavelength polynomial | frequency capped at `2.463e15 Hz` | none |
| He I Rayleigh | same | actual He I | analytic polarizability fit | frequency capped at `5.15e15 Hz` | none |
| H2 Rayleigh | same | locally reconstructed H2 | analytic inverse-wavelength polynomial | frequency capped at `2.922e15 Hz`; H2 zero for `T>20000 K`; requires IFOP 4 and 13 | none |
| Rosseland surrogate | `compute_rosseland_continuum_opacity_columns` | `T`, gas pressure | nearest-quadrant table interpolation | only IFOP 19 with supplied table; empty table returns `1.0` | `4 sigma T^4 / 4pi`-style scalar |

The Rosseland surrogate is listed for completeness. It is disabled by default
and is not the physical continuum construction that Chapter 5 should use as
its teaching path. Its frequency-independent source expression is bolometric
rather than \(B_\nu\)-per-Hz, so mixing it into the ordinary source numerator
is a legacy optional convention, not a unit-consistent standard continuum
branch.

## 7. Atmosphere-local H2 boundary

The continuum path deliberately reconstructs H2 instead of reading a stored H2
population:

```text
neutral H actual density
    -> recompute six-level H I partition
    -> normalized H I ground population
    -> multiply by 2 and ground departure coefficient
    -> square
    -> multiply by atmosphere H2 equilibrium constant
    -> H2 density for CIA and Rayleigh only
```

`_h2_equilibrium_constant(...)`:

- replaces nonfinite or `T<=100 K` by `100 K`;
- clips the interpolation temperature to `19900 K`;
- uses table indices 1–199 and the exact adjacent-entry formula;
- returns zero where its final value is nonfinite.

The consumers then replace H2 by zero only where `T>20000 K`. Thus the
`19900–20000 K` interval uses the 19900 K-clipped constant, and the value at
exactly 20000 K is retained.

This is related to, but not identical with:

- the Chapter 4 atmosphere molecular-equilibrium H2 branch;
- the synthesis H2 Rayleigh reconstruction, which uses a clamped polynomial
  exponential rather than this partition table;
- the schema-v4 `molecular_hydrogen_population` field.

`AtmosphereConfig.enable_molecules` controls whether the Chapter 4 packed CH
and OH slots are populated. It does not switch off the IFOP-9 continuum call
and does not switch off the local H2 reconstruction. With molecules disabled,
CH/OH are therefore normally zero while H2 CIA can remain active. Record this
as a configuration/consumer boundary rather than calling all three terms
jointly enabled or disabled.

The chapter should teach the physics once, then show these as three exact
consumer policies in one compact comparison. It must not imply that all three
read a single H2 array.

## 8. Sampling and interpolation contract

### 8.1 The direct atmosphere grid

`build_opacity_sampling_grid(effective_temperature)` always returns 30,000
float64 wavelengths and 30,000 float64 frequency weights. With one-based
sample index \(i\),

\[
\lambda_i[{\rm nm}]
=10^{1+10^{-4}(i+s-1)}.
\]

The start offset `s` is selected by ordered temperature thresholds:

| Effective-temperature condition | `start_index` | First wavelength |
|---|---:|---:|
| `T_eff < 4500 K` | 11601 | 144.577263387 nm |
| `4500 <= T_eff < 7250 K` | 9599 | 91.1800865275 nm |
| `7250 <= T_eff < 13000 K` | 7027 | 50.4312810266 nm |
| `13000 <= T_eff < 30000 K` | 3577 | 22.7876741143 nm |
| `T_eff >= 30000 K` | 1 | 10.0023028502 nm |

The inequalities are exact; a fixture must hit both sides of 4500, 7250,
13000, and 30000 K.

Interior frequency weights are centered half differences in frequency. The
first and final weights use distinct pinned endpoint formulas. They must be
captured rather than replaced by a generic trapezoid helper.

The production caller computes
`opacity_frequency_hz = 2.99792458e17 / opacity_wavelength_grid_nm` and
evaluates every enabled process directly at all 30,000 frequencies. There is
no atmosphere edge-triplet interpolation to an arbitrary wavelength grid.

### 8.2 The line-selection reference continuum

`build_continuum_reference_wavelength_grid()` contains 343 literal wavelength
points from 9.09 to 400000 nm, followed by a duplicated 344th wavelength. Its
packed coordinate is

```text
int(log(wavelength_nm) / log(1 + 1/2_000_000) + 0.5)
```

for the first 343 values; the final packed value is the sentinel `2**30`.

Only reference wavelengths strictly greater than the first active 30,000-grid
wavelength are evaluated. The number of active points is 226, 240, 263, 299,
or 338 in the five temperature regimes above.

`assemble_continuum_line_selection_threshold(...)` writes a `(D,344)`
float32 table:

\[
q_\nu =
\frac{10^{-3}(\kappa_\nu^{\rm abs}+\sigma_\nu)}
     {\max[1-\exp(-h\nu/kT),10^{-300}]}.
\]

Inactive short-wavelength entries use `1e10` in place of the opacity before
the same `1e-3 / stimulated_emission` scaling. Column 343 duplicates column
342. This is a line-selection threshold with stimulated emission divided out,
not a raw continuum-opacity archive.

### 8.3 Interpolation internal to individual processes

The atmosphere path still has local table interpolation:

- Karzas-Latter: log-frequency interpolation, threshold zero, high-shell
  extension;
- H-minus bound-free: the exact piecewise-quadratic remap imported from
  `microturbulence.py`;
- H-minus free-free: linear interpolation/extrapolation in log wavelength and
  linear interpolation/extrapolation in theta;
- CH/OH: bilinear interpolation over photon energy and temperature, plus
  partition interpolation;
- H2 CIA: wavenumber and temperature table interpolation;
- Si II: generated table interpolation;
- Rosseland surrogate: nearest point in each normalized `(log T, log P)`
  quadrant, followed by bilinear-like interpolation or inverse-distance
  fallback.

These local operations are not substitutes for synthesis edge-triplet
interpolation.

## 9. CPU, Numba, dtype, and ordering contract

Numba is a hard dependency. Although some source functions retain NumPy or
Python fallback text, import failure raises before the module can be used and
`_NUMBA_AVAILABLE` is always true in the production path.

Every continuum table and state array is converted to host float64. The
atmosphere continuum has no Torch, CUDA, or MPS path.

| Kernel | Decorator | Parallel axis | Independence argument |
|---|---|---|---|
| `_planck_frequency_exact_kernel` | `njit(parallel=True, nogil=True, cache=True)` | frequency | each frequency writes one complete depth column |
| `_coulomb_freefree_gaunt_kernel` | same | frequency | each output `(depth,frequency)` cell is independent |
| `_linear_interpolate_kernel_serial` | `njit(cache=True, nogil=True)` | serial query order | used below 8192 queries to avoid dispatch overhead |
| `_linear_interpolate_kernel_parallel` | parallel `njit` | query point | each query independently finds its bracket |
| `_iron_neutral_branch_kernel` | parallel `njit` | frequency | 48-transition sum stays local to one output column |
| `_helium_low_level_grid_kernel` | parallel `njit` | frequency | five low-level values write one independent column |
| `_karzas_latter_cross_section_grid_kernel` | parallel `njit` | frequency | pure cross-section evaluation per frequency |
| `_lukewarm_metal_absorption_kernel` | parallel `njit` | frequency | each frequency owns its depth column; dot products are local |
| `_evaluate_rosseland_opacity_kernel` | serial `njit` | none | one `(T,P)` lookup |

`NUMBA_NUM_THREADS` controls the parallel kernels. The source configures a
persistent cache through `NUMBA_CACHE_DIR` or
`PAYNE_ZERO_NUMBA_CACHE_DIR`. Cold compilation, warm cached execution, and
one-thread/many-thread execution must be measured separately.

The hot-metal composite remains an ordered Python loop over frequency chunks
of 4096 and ion charges 1–5; its Gaunt-grid calls are internally parallel.
The H-minus theta loop, process sum order, opacity-flag order, and the two
separate production continuum calls remain ordered.

## 10. Exact edge and failure policies

These policies belong in tests even when only the physically ordinary path is
shown in the notebook.

### 10.1 Data and state

- Missing archives or missing required keys raise
  `ContinuumOpacityTableError`.
- Continuum, Karzas, and H2 table shapes are checked exactly.
- Loaded numeric fields are cast to float64.
- `ContinuumLevelTables` has no shape validation, but it is inactive.
- Packed state depth mismatch, fewer than two actual population slots, or no
  normalized slot 847 raises `ValueError`.
- Missing hydrogen departure coefficients become `(D,6)` ones.
- An existing departure array is checked only on its depth axis. The H I
  component pads fewer than six columns with ones; the state adapter itself
  does not enforce a six-column minimum.

### 10.2 Frequency and thermodynamic domain

- Public component functions require a one-dimensional frequency array.
- They do not require positive, finite, sorted, or nonempty frequencies.
- Many formulas take `log(frequency)` or divide by powers of frequency.
  Invalid frequencies can therefore produce warnings, NaN, or infinity
  rather than a clean exception.
- Mass density is generally floored at `1e-300`; some synthesis analogues use
  `1e-30`, so cross-backend out-of-domain behavior is not interchangeable.
- Temperature is floored locally in many formulas but not uniformly. H-minus
  theta and He-minus contain direct division by temperature.
- Final component and composite opacity arrays are not globally clipped.

The textbook implementation should validate the physical fixture domain before
calling the exact kernels. It should retain the exact kernel behavior beneath
that validation and test that valid inputs never need a repair clamp.

### 10.3 Branch-specific boundaries

- Flag values other than exact integer `1` are off.
- A short flag vector is padded with zeros.
- IFOP 9 switches molecular and five neutral-metal families together.
- H2 Rayleigh cannot be enabled by IFOP 13 alone because its H I ground
  population is created only inside IFOP 4.
- CH and OH are zero per layer at `T>=9000 K`.
- The molecular composite is skipped entirely only when every layer is at
  least 9000 K. If any layer is cooler, H2 CIA is evaluated for every layer
  and is then limited only by its own 20000 K policy.
- H2 CIA is zero beyond `20000 cm^-1`.
- H2 is zero for `T>20000 K`, not at exactly 20000 K.
- H I, He I, and H2 Rayleigh cross sections cap the input frequency before
  converting to wavelength; they become constant above their cap.
- The hot-metal bound-free pruning threshold is path-dependent because it is
  compared with the running free-free column.
- An empty Rosseland table evaluates to `1.0`, but the default IFOP 19 is off.

## 11. Atmosphere–synthesis reconciliation

The two implementations must be presented as two exact consumers of shared
physics, not forced into false bitwise identity.

| Topic | Atmosphere | Synthesis |
|---|---|---|
| Hardware | NumPy + Numba multicore CPU | Torch on CUDA/MPS/CPU |
| Main grid | direct 30,000-point Teff-dependent grid | only used three-sample edge intervals, interpolated to `(D,W)` |
| Work dtype | float64 | float64 on CUDA/CPU, float32 on MPS |
| Continuum source | absorption-weighted source; H/H-minus departure support | Planck source returned by `compute_sampled_continuum` |
| Feature flags | external 20-entry IFOP vector | standard composite has no matching IFOP surface |
| CH/OH/H2 CIA | included under IFOP 9 | absent from standard `continuum(...)` |
| H2 Rayleigh density | atmosphere H2 partition table, with 19900/20000 K policy | separate polynomial exponential clamped to exponent `[-100,100]` |
| H I Rayleigh | literal inverse-wavelength fit | Gavrila/polarizability tables |
| Si II | generated cached `(200,51)` atmosphere table | packaged Peach tables |
| Edge metadata | not used by atmosphere continuum | required schema-v4 input |
| `FrequencyInvariants` | no such object | available, but standard pipeline does not pass it to `continuum(...)` |

Overlapping processes should receive independent atmosphere-oracle and
synthesis-oracle checks. A cross-lane equality test is appropriate only for a
deliberately restricted state and formula shown to be common. CH/OH/CIA
presence in one lane and absence in the other is an expected feature test, not
a parity failure.

## 12. Causal teaching order

Chapter 5 should retain two movements and one final continuum artifact.

### Movement 5A — From one cross section to a light-element continuum

1. Begin with a line-free spectral interval that still has wavelength-dependent
   extinction.
2. Convert one absorber cross section to `n * sigma / rho`, with units.
3. Reuse, rather than rederive, the Chapter 1 Planck function and Chapter 3
   actual-versus-normalized populations.
4. Derive the stimulated-emission factor and check both limits.
5. Build H-minus bound-free and free-free opacity at one depth. Let the
   threshold and the solar-regime budget motivate interpolation tables.
6. Build H I bound-free plus H II free-free. Introduce explicit levels, the
   high-level tail, and the Coulomb Gaunt factor only when each is needed.
7. Add H2-plus and He-minus as compact fitted processes.
8. Add He I and He II, comparing a solar and hot layer.
9. Assemble H/He absorption while keeping each named budget visible.

The movement closes with checked light-element absorption and source columns,
not with the full continuum.

### Movement 5B — Metals, molecular continua, scattering, and two grids

1. Show a residual near-UV budget that H/He alone cannot explain.
2. Add C I, Mg I, Al I, Si I, and Fe I as named neutral-metal processes.
3. Add the lukewarm and hot groups while showing once why bound-state terms use
   normalized populations and ionic free-free uses actual charge-squared
   populations.
4. Introduce CH, OH, and H2 CIA through a cool atmosphere. State immediately
   that these are atmosphere-only in the standard composite.
5. Introduce Thomson scattering, then H I/He I/H2 Rayleigh scattering. Keep
   absorption and scattering in separate arrays.
6. Build the direct 30,000-point atmosphere grid, including the four exact
   Teff thresholds.
7. Build the smaller 343/344-point continuum used only to decide which lines
   can matter.
8. Contrast this CPU grid with the synthesis edge-triplet construction. Derive
   the three interpolation basis functions only in the synthesis lane.
9. End with per-process and ordered-sum parity in all four regimes.

The useful parameter variations—one population doubling, one density scaling,
one threshold crossing, one hot/cool comparison, one scattering color trend,
and one grid-boundary comparison—belong inline at the moment they answer a
live question. There is no detached exercise section.

## 13. Required staged fixtures and oracle captures

### 13.1 Static inputs

Stage, with manifest role `static_physical_input`:

- the three active atmosphere archives in section 3.1;
- all required archive keys, including loader-required inactive fields;
- exact source commit, source path, SHA-256, array shape, dtype, unit, and
  consumer routine;
- a compact human-readable excerpt for H-minus, Gaunt, CH/OH, H2 CIA, and hot
  transitions used in plots.

Do not stage `continuum_level_tables.npz` as an active continuum dependency.
If retained for package completeness, label it `inactive_exported_table`.

### 13.2 State fixtures

Use the four checksum-bound regimes:

| Fixture | Required active evidence | Required near-zero evidence |
|---|---|---|
| `hot_dwarf` | He II/He III, ionic free-free, Thomson, hot metals | weak H-minus and molecular continua |
| `solar_dwarf` | H-minus, H I, ordinary neutral metals, H I Rayleigh | weak hot-ion extremes |
| `low_gravity_giant` | density-per-mass and scattering response | pressure-sensitive small terms remain finite |
| `cool_molecule_rich` | CH, OH, H2 CIA, H2 Rayleigh, neutral metals | hot-ion continua |

Each compact atmosphere fixture must contain the exact
`ContinuumAtmosphereState` fields and the upstream schema/runtime identity from
which they were derived. Include at least six depths spanning outer, middle,
and deep layers; full integration captures retain the native depth count.

Add isolated synthetic state fixtures for:

- normalized-population doubling;
- actual-ion charge-square sum;
- fixed `n_e/rho` Thomson constancy;
- H2 reconstruction around 8999/9000, 19900, and 20000 K;
- IFOP 4/13 coupling;
- non-LTE H and H-minus source behavior;
- molecules-enabled and molecules-disabled columns showing CH/OH versus
  continuum-local H2 behavior.

### 13.3 Frequency fixtures

One compact ordered frequency vector must include values immediately on both
sides of:

- the H-minus bound-free edge `1.82365e14 Hz`;
- H2 CIA `20000 cm^-1`;
- Fe I `21000 cm^-1`;
- the H2-plus/Lyman value `3.28805e15 Hz`;
- one He I low-level threshold;
- the H I/He I/H2 Rayleigh frequency caps;
- selected C/Mg/Al/Si edges from the active tables.

Capture the full 30,000-point grid separately for the five Teff start regimes,
including exact threshold temperatures.

### 13.4 Golden outputs

Golden files are comparison-only and must be opened after the textbook
calculation. Capture:

1. every named component in section 6 on the compact frequency/depth fixture;
2. the ordered absorption sum after each IFOP group;
3. scattering split into electron, H I, He I, and H2;
4. source numerators for H I, H-minus, and the LTE aggregate;
5. final `(D,F)` absorption, scattering, and source;
6. 30,000-grid wavelength, frequency, and weights;
7. active reference indices/frequencies and `(D,344)` float32 line threshold;
8. complete `OpacityState` continuum fields for all four regimes;
9. cold-cache and warm-cache outputs;
10. one-thread and available many-thread outputs.

Every capture records source/data hashes, flags, effective temperature, Numba
and NumPy versions, CPU, thread count, cache state, dtype, and tolerance.

## 14. Mandatory parity and science gates

### 14.1 Source and data

- pinned commit and every SHA-256 above match;
- every required table key, shape, and dtype matches;
- active versus loader-required-inactive table fields are asserted;
- no runtime import reads the pinned source tree.

### 14.2 Local physics

- `n * sigma / rho` reduces to cm2 g^-1;
- stimulated emission lies in `[0,1]` for valid inputs and reaches both limits;
- each bound-free term is zero on its inactive side;
- isolated opacity doubles with its absorber population;
- isolated mass opacity halves when only mass density doubles;
- Coulomb free-free responds to charge squared;
- Thomson scattering is frequency independent at fixed `n_e/rho`;
- Rayleigh terms increase strongly blueward below their caps and flatten above
  the exact cap;
- the H2 bridge reproduces all endpoint and cutoff policies;
- CH/OH vanish per layer at 9000 K;
- no valid-domain fixture needs a repair clamp.

### 14.3 Numerical and execution parity

- teaching scalar Planck, Gaunt, Karzas, He-low-level, Fe, lukewarm, and
  interpolation helpers match their compiled production kernels;
- serial and parallel LINTER agree below and above the 8192-query dispatch
  boundary;
- one-thread and many-thread results agree under a declared ULP/tolerance
  policy;
- cold and warm cache results are identical;
- ordered component sums match the pinned source before a total can hide a
  branch error;
- each IFOP switch changes only its owned group, including the intentional
  IFOP 4/13 coupling and IFOP 9 grouping;
- the final three arrays match in every regime;
- line-selection thresholds match bitwise where float32 conversion makes that
  reasonable, otherwise under an explicitly measured tolerance.

### 14.4 Grid and boundary parity

- all five start offsets and exact threshold inequalities match;
- the first, interior, and final frequency-weight formulas match;
- the reference grid has 343 physical values plus the duplicated/sentinel
  entry;
- active reference counts are exactly 226, 240, 263, 299, and 338 in the five
  start regimes;
- the production caller performs two continuum evaluations: full sampling and
  active line-reference frequencies;
- a call trace confirms no synthesis edge-triplet routine is used by the
  atmosphere path;
- a feature test confirms CH/OH/H2 CIA are present in the atmosphere composite
  and absent from synthesis `continuum(...)`.

## 15. P0 ambiguities to resolve before implementation

1. **H2 CIA temperature interpolation order.** At
   `continuum_opacity.py:5166–5175`, the lower table column is multiplied by
   `temperature_fraction` and the upper column by `1-temperature_fraction`.
   At an exact lower grid temperature this selects the upper column. Capture
   pinned oracle values at exact grid nodes and one midpoint before deciding
   whether the textbook exposes this as a parity-pinned convention or a source
   defect. Do not silently reverse it.
2. **IFOP 13 depends on IFOP 4.** H2 Rayleigh is skipped if H I Rayleigh is
   disabled because the needed reconstructed H I ground population is local
   to the IFOP 4 branch. Verify with an oracle and document the combined switch
   semantics; do not present IFOP 13 as independently functional.
3. **Final non-negativity claim.** `BIBLE.md` and the chapter contract request
   nonnegative slabs, but the exact source does not clamp and accepts
   out-of-domain frequency/temperature inputs. Define the supported physical
   input preconditions and prove non-negativity there. Any public validation
   wrapper must fail before the exact kernel rather than mutate its result.
4. **H2 field ownership.** The brief currently says Chapter 4 provides an H2
   population, but the atmosphere continuum recomputes it. The chapter must
   distinguish the schema field, atmosphere molecular solver output, atmosphere
   continuum-local reconstruction, and synthesis polynomial reconstruction.
   It must also show that `enable_molecules=False` zeros CH/OH production but
   does not disable continuum-local H2.
5. **Inactive but loader-required table fields.** Gavrila H I Rayleigh and
   Peach Si II arrays are mandatory keys in the atmosphere archive but unused
   by atmosphere formulas. The manifest needs two roles: `required_for_loader`
   and `consumed_by_process`. Otherwise coverage audits will falsely claim that
   the atmosphere executes those tables.
6. **Inactive `ContinuumLevelTables`.** The exported loader and archive are
   dead in the pinned composite. Exclude them from the minimal teaching path
   unless an oracle trace proves a live caller.
7. **Cross-lane parity wording.** Atmosphere H I Rayleigh, Si II, H2 density,
   source behavior, floors, flags, and grids differ from synthesis. Exact
   parity means matching each pinned lane, not requiring the two lanes to
   match one another.
8. **Rosseland surrogate ownership.** IFOP 19 can add an interpolated
   Rosseland opacity as though it were a continuum component, but it is default
   off and Chapter 11 owns Rosseland means. Keep the routine in the exact
   coverage matrix and tests, while deferring its physical derivation and
   excluding it from the standard Chapter 5 sum.
9. **`opacity_flags=None` is not the runner default.** All chapter calls must
   pass the explicit default vector or a named isolated vector. Otherwise the
   optional IFOP 19 branch can be accidentally activated when a table is
   supplied.
10. **H-minus archive-unit spelling.** The key
    `hminus_boundfree_cross_section_cm2` stores order-unity values and both
    atmosphere and synthesis multiply by `1e-18`. The manifest must declare
    the stored unit as \(10^{-18}\ {\rm cm^2}\) while retaining the exact key.
11. **Rosseland surrogate source units.** IFOP 19 combines a bolometric
    temperature-fourth source with per-frequency \(B_\nu\) sources. Preserve
    this only in a separately labeled optional-branch parity test; do not
    activate it in the standard physical continuum or use it to teach source
    units.
12. **Column-global molecular gate.** The single
    `np.min(temperature) < 9000` condition controls whether the molecular
    composite is entered. CH/OH still zero warm layers locally, but H2 CIA is
    then evaluated in warm layers up to its separate 20000 K cutoff. Capture
    an all-warm column and a mixed cool/warm column before explaining this
    exact atmosphere-column policy.

No Chapter 5 coding should begin until items 1–5 and 10–12 have oracle captures
and an explicit decision recorded in the chapter contract.

## 16. Completion contract

The atmosphere-continuum portion of Chapter 5 is complete only when a reader
can:

1. identify the actual or normalized population used by every process;
2. compute each named mass-opacity component from a supplied Chapter 4 state;
3. keep absorption, scattering, and thermal source distinct;
4. reproduce the exact IFOP group behavior;
5. build the direct 30,000-point atmosphere grid and line-reference threshold;
6. explain where and why Numba parallelism is safe;
7. reproduce all four pinned atmosphere regimes without importing Payne Zero;
8. state precisely why atmosphere CH/OH/H2 CIA, H2 policy, and sampling differ
   from the synthesis continuum path.

The causal handoff to Chapter 6 is then clean: the smooth background and its
line-selection threshold are trustworthy, but no bound-bound transition has
yet been given a profile.

## 17. Concise author handoff

### 17.1 Causal teaching order

```text
cross section
  -> absorber number density
  -> mass absorption
  -> stimulated emission and source
  -> H-minus
  -> H I / H II
  -> H2-plus and helium families
  -> neutral, lukewarm, and hot metals
  -> atmosphere-only CH / OH / H2 CIA
  -> electron and Rayleigh scattering
  -> direct atmosphere sampling grid
  -> line-selection reference continuum
  -> synthesis edge-triplet contrast
  -> component and full-slab parity
```

### 17.2 Minimal visible-code checkpoints

These ten are the minimum reader-visible scientific checkpoints. Exact long
kernels remain in the progressive package and are called here; any additional
visible cell must resolve a causal step that cannot stay readable inside one
of these checkpoints.

| Checkpoint | Visible computation and immediate output |
|---:|---|
| 1 | Evaluate `n_absorber * sigma / mass_density` and the stimulated-emission factor at two limits; print units and limits |
| 2 | Verify manifest hashes, load `ContinuumOpacityTables` and `KarzasLatterTables`, and print only key shape/role facts |
| 3 | Evaluate H-minus bound-free/free-free on both sides of its edge; one one-panel component plot |
| 4 | Call the exact H I, H2-plus, He-minus, He I, and He II components for solar and hot depths; one ordered budget plot |
| 5 | Perturb normalized and actual populations independently, then call neutral, lukewarm, and hot metal components; print the two ownership checks |
| 6 | Call CH/OH/H2 CIA around their temperature/frequency boundaries and compare the three H2 consumer policies inline |
| 7 | Isolate Thomson and H I/He I/H2 Rayleigh scattering; print the constant and blueward-limit checks |
| 8 | Build all five exact 30,000-point atmosphere grid regimes; print start index, first wavelength, and endpoint-weight checks |
| 9 | Build the 343/344-point line-selection threshold; print active count, dtype, duplicated final column, and sentinel |
| 10 | Assemble named components and final slabs for the four regimes, open goldens only afterward, and print a compact parity table |

The synthesis three-basis interpolation can share checkpoint 9 as a short
contrast calculation. It must not become a second atmosphere-grid
implementation.

### 17.3 Provenance and data files

The minimal active atmosphere bundle is:

```text
payne_zero_atmosphere/continuum_opacity.py
payne_zero_atmosphere/constants.py
payne_zero_atmosphere/population_layout.py
payne_zero_atmosphere/runtime_state.py
payne_zero_atmosphere/runner.py
atmosphere_tables/continuum_opacity_tables.npz
atmosphere_tables/karzas_latter_tables.npz
atmosphere_tables/molecular_equilibrium_tables.npz
```

Copy only into textbook-owned static/fixture/golden roles, retain the hashes
in sections 2–3, and never import the pinned tree at reader runtime.
`continuum_level_tables.npz` is not active. Synthesis additionally owns its
separate `continuum_tables.npz` and `continuum_edge_grid.npz`; those are not
substitutes for the atmosphere archives.

### 17.4 Explicit atmosphere-versus-synthesis non-equivalences

- direct 30,000-point CPU sampling is not edge-triplet interpolation;
- float64 NumPy/Numba state is not device work-dtype Torch state;
- departure-aware H/H-minus sources are not the synthesis Planck-only
  continuum source;
- atmosphere CH/OH/H2 CIA are not standard synthesis terms;
- atmosphere table-based H2 reconstruction is not the synthesis polynomial
  reconstruction and neither is the stored schema H2 field;
- atmosphere analytic H I Rayleigh is not synthesis Gavrila Rayleigh;
- atmosphere generated Si II is not synthesis Peach-table Si II;
- atmosphere IFOP groups are not synthesis feature switches;
- matching each pinned lane is exact parity; forcing the lanes to equal one
  another is not.
