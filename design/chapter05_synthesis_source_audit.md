# Chapter 5 Exact Synthesis-Continuum Source Audit

Status: source-complete audit for the pinned Payne Zero tree. This document is
an implementation and teaching contract, not reader-facing prose and not a
source tour.

## 1. Audit decision

Chapter 5 can be built as one causal chapter with two movements, but it must
distinguish three continuum lanes that the pinned source keeps separate:

1. **The standard structured-synthesis lane.** `SynthesisPipeline.run()` calls
   `continuum(...)`. That routine chooses only the edge intervals used by the
   requested wavelength grid, evaluates three frequencies per used interval,
   and log-parabolically interpolates the results. It calls
   `_compute_at_freqs(...)` without `FrequencyInvariants` and with the default
   `coulomb_table_energy_first=False`.
2. **The sampled diagnostic lane.** `compute_sampled_continuum(...)` evaluates
   caller-supplied frequencies, returns absorption, scattering, and the LTE
   Planck source, and forces `coulomb_table_energy_first=True`. With no
   `FrequencyInvariants`, it remains a per-frequency fallback.
3. **The explicitly precomputed sampled lane.**
   `build_frequency_invariants(...)` plus
   `compute_sampled_continuum(..., frequency_invariants=...)` activates the
   fully materialized `(depth, frequency)` helpers, including the explicit
   light-element grid. This is implemented and worth teaching as an
   independently verified extension, but the pinned standard pipeline neither
   builds nor passes this object.

The chapter must make the first lane the parity destination. The second lane is
useful for component experiments and compact goldens. The third lane belongs in
an honestly labeled performance/architecture boundary. Treating all three as
interchangeable would teach physics that the production call does not execute.

The chapter must also state a second boundary without euphemism:

- CH bound-free absorption, OH bound-free absorption, and H2
  collision-induced absorption (H2-H2 plus H2-He) are active atmosphere-package
  continuum processes.
- They are absent from `payne_zero_synthesis.continuum.continuum(...)`.
- H2+ absorption and H2 Rayleigh scattering are nevertheless present in the
  synthesis continuum. These are different processes.

This agrees with `BIBLE.md`, `PLAN.md`, and
`design/global_chapter_contracts.md`, while refining two overly broad claims in
`design/part3_part4_synthesis_brief.md`: the complete
`FrequencyInvariants` lane is not the standard lane, and the synthesis
continuum does not consume the schema's stored
`molecular_hydrogen_population`.

## 2. Frozen evidence and source identity

Read-only source root:
`/Users/ysting/payne-zero`

Pinned commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

The audited working-tree blobs match the pinned commit blobs:

| Source | Git blob | SHA-256 of checked-out bytes | Primary anchors |
| --- | --- | --- | --- |
| `payne_zero_synthesis/continuum.py` | `e197bb648a1c890c30bca65aaf57dba95231e7e3` | `ab0d4eb771ee04101f6936253f633ed60d845e2816854a06b1b059e8b91dce1b` | tables `476`; populations `1692`; H- `1936`; H I `2081`; scattering `2298`; minor terms `3538`; He `4144`; hot/Si II `4563`; light elements `4711`; invariants `4906`; sampled driver `5131`; triplets `5464`; public composite `5515` |
| `payne_zero_synthesis/pipeline.py` | `31a30e1b0b1dc76d8e78f475410e412d634cc8d9` | `465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22` | edge table load `277`; schema construction `320`; continuum view `1055`; standard call `1315`; table upload `1631` |
| `payne_zero_synthesis/synthesis.py` | `4878f4540bafaf464adca62cd64fc5adba914bf6` | `590e430b6582fbcf601a52b721d8f65073432903773a99238073a1d821fe0d0c` | runtime resolution and structured builder `29`; synthesis boundary `75` |
| `payne_zero_synthesis/equation_of_state.py` | `e40114e2e1b15e834ccfeaa151783954b92d8ca7` | `6497c29abb954e0b55d918cc22fa7b660952812c548faf1d7b1053345ef13562` | upstream actual and partition-normalized population ownership |
| `payne_zero_synthesis/constants.py` | `c9a34cde1b37325a15c1e88354c5027425d88bc3` | `ed58004196790f9fb4a2871044c9cd36bf7bc42046a923f9314f7b8ea7456798` | exact and parity-rounded constant tiers `1` |
| `payne_zero_synthesis/device.py` | `99ad9c8a18a26d90dbc7fe46da72eb6dc958332d` | `22e769ebed60ad3a0f2060264247e469a99afd20ec5cadb69a01b6e5fa82ea3c` | device and dtype policy `1` |
| `payne_zero_synthesis/paths.py` | `830f183371c55c4fa3ea3d515adad3ecf269c10b` | `2bca3284eb1765449ab3fc87439eb603e3941213d9c1205c71aee3fd1ad30b5d` | `PAYNE_ZERO_DATA_ROOT` and `SYNTHESIS_TABLE_DIR` resolution `23` |
| `payne_zero_synthesis/api.py` | `0ea3a924f6b5a7b32d13828d57f9b23ffa56290f` | `77718303c1e0052a520ece7fab277b3b1922c21d09b35a288596592d03310940` | structured builder `344`; public `synthesize` to engine boundary `442` |
| `payne_zero_synthesis/atmosphere.py` | `c4c0283cc6271708146f1e47c7e17fae8ab5a29e` | `06b79770e4d9472093655022d53ee7fddf7cc6727206f34c0f60c57151e2cf9b` | schema validation, including edge arrays and population roles `470` |
| `payne_zero_synthesis/atmosphere_schema.json` | `7af171bd587fe3afe2700c3ec8525dd8932fd3f2` | `2ba8d637e613be12ff43ce319a752616323f0341ea69f8e2391c3c244939777a` | schema-v4 units and notes |
| `payne_zero_atmosphere/continuum_opacity.py` | `6c805f73644c10a79a37f210849d6bda9649d31e` | `1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81` | H2 population `1297`; CH `4955`; OH `5029`; CIA `5103`; molecular composite `5188`; atmosphere composite `5834`; scattering `6056` |
| `payne_zero_atmosphere/synthesis_bridge.py` | `fe535efe9bfcef0bd3d5fd6e55cbbec21c480c34` | `142a960b5e710823754b02766803b3c1dd8c48c9945fdfabe560b4ee7e1acb50` | edge-grid schema handoff `174` |

The required LFS payloads also match the object IDs at the pinned commit:

| Physical input | LFS SHA-256 / checked-out SHA-256 | Bytes | Role |
| --- | --- | ---: | --- |
| `source_data_files/synthesis_tables/continuum_tables.npz` | `406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d` | 34,516 | synthesis cross-sections, Gaunt factors, H- tables, Rayleigh tables |
| `source_data_files/synthesis_tables/continuum_edge_grid.npz` | `11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e` | 24,104 | signed edges, wavelengths, triplet samples, interpolation geometry |
| `source_data_files/atmosphere_tables/continuum_opacity_tables.npz` | `6fd4c556418870c28d3fcc9a050252af58ac4cc433cae979477355c8c7d593e3` | 57,456 | atmosphere continuum, including CH/OH/CIA tables |
| `source_data_files/atmosphere_tables/molecular_equilibrium_tables.npz` | `1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe` | 1,935 | atmosphere H2 partition table used by Rayleigh and CIA |

No source or paper file was modified during this audit.

## 3. Exact reader-facing continuum contract

Use these symbols only in explanatory derivations; introduce the exact names at
the implementation boundary.

| Quantity | Exact name | Shape / axis order | Unit | Residency |
| --- | --- | --- | --- | --- |
| requested synthesis wavelength | `wavelength_grid_nm` | `(W,)`, increasing in the standard grid | nm | host float64 |
| sampled physical frequency | `frequencies_hz` | `(F,)` | Hz | host float64; small work-dtype tensor when needed |
| layer temperature | `temperature` | `(D,)`, outermost to innermost | K | work-dtype device tensor plus host float64 copy |
| mass density | `mass_density` | `(D,)` | g cm^-3 | work-dtype device tensor |
| absorption mass opacity | `continuum_absorption` | `(D,F)` sampled, `(D,W)` interpolated | cm^2 g^-1 | work-dtype device tensor |
| scattering mass opacity | `continuum_scattering` | `(D,F)` sampled, `(D,W)` interpolated | cm^2 g^-1 | work-dtype device tensor |
| LTE continuum source | `continuum_source` | `(D,F)` | Planck `B_nu`, cgs per Hz per sr | work-dtype device tensor |
| edge frequency | `signed_continuum_edge_frequency_hz` | `(341,)` | signed Hz; runtime uses magnitude | host float64 |
| edge wavelength | `continuum_edge_wavelength_nm` | `(341,)`, strictly increasing | nm | host float64 |
| edge midpoint | `continuum_edge_midpoint_wavelength_nm` | `(340,)` | nm | host float64 |
| quadratic denominator | `continuum_edge_interval_width_squared_over_two_nm2` | `(340,)` | nm^2 | host float64 |

The construction unit identity is

\[
  \kappa_\nu
  = \frac{n_{\rm absorber}\,\sigma_\nu}{\rho},
  \qquad
  \frac{{\rm cm}^{-3}{\rm cm}^{2}}{{\rm g\,cm}^{-3}}
  = {\rm cm}^{2}\,{\rm g}^{-1}.
\]

For LTE absorptive processes, the recurring stimulated-emission factor is

\[
  s_\nu(T)=1-\exp\!\left(-\frac{h\nu}{kT}\right).
\]

The exact source sometimes folds this factor into a fitted coefficient or
applies a branch-specific algebraic equivalent. The chapter must derive the
factor once and then show where each routine actually applies it; it must not
blindly multiply every process by a second copy.

`compute_sampled_continuum(...)` returns `continuum_source = B_nu(T)`.
`continuum(...)`, the standard pipeline entry, returns only absorption and
scattering. Chapter 9 owns how those slabs enter the transfer source.

## 4. Actual call graph and lane ownership

```text
SynthesisPipeline.run
  -> continuum(wavelength_grid_nm, trimmed_schema_view, ContinuumTables)
       -> build_pops(...)                         [unless explicitly supplied]
       -> build_edge_sample_frequencies(all 340 intervals)
       -> searchsorted + unique used intervals
       -> _compute_at_freqs(three samples per used interval)
            frequency_invariants = None
            coulomb_table_energy_first = False
       -> log10(max(opacity, 1e-30))
       -> three-point wavelength interpolation
       -> 10**log_opacity
  -> (continuum_absorption, continuum_scattering), both (D,W)
```

The standard call does **not** call `compute_sampled_continuum`, does **not**
call `build_frequency_invariants`, and does **not** store a
`FrequencyInvariants` object inside pipeline `WindowInvariants`.

The three lanes differ as follows:

| Lane | Entry | Coulomb table layout flag | Full `(D,F)` materialization | Explicit N I/O I/Mg II/Ca II grid | Pipeline-owned |
| --- | --- | --- | --- | --- | --- |
| edge-triplet standard | `continuum(...)` | `False` | no | no | yes |
| sampled fallback | `compute_sampled_continuum(..., frequency_invariants=None)` | `True` | no | no | no |
| sampled precomputed extension | `compute_sampled_continuum(..., frequency_invariants=build_frequency_invariants(..., True))` | `True` | yes | yes | no |

The same `coulomb_freefree_gaunt_table (12,11)` is interpreted through two
validated layouts. The flag is therefore scientific state, not a disposable
performance toggle.

The exact `FrequencyInvariants` fields are:

| Field group | Exact fields and shapes |
| --- | --- |
| identity | `frequencies_hz (F,)`; `coulomb_table_energy_first` scalar bool; `natural_log_frequency (F,)` |
| H- and Rayleigh | `hminus_freefree_rows (F,11)`; `hminus_boundfree_cross_section_cm2 (F,)`; `rayleigh_factor (F,)` |
| Si II bracket state | `silicon_singly_ionized_peach_frequency_rows (F,6)`; `silicon_singly_ionized_peach_natural_log_temperature_grid (6,)` |
| H I photoionization | `hydrogen_high_level_photoionization_cross_sections (9,F)`; `hydrogen_low_level_photoionization_cross_sections (5,F)`; `hydrogen_ground_level_photoionization_cross_section (F,)`; `hydrogen_tail_edge (F,)` |
| He I photoionization | `neutral_helium_low_level_photoionization_cross_sections (10,F)`; `neutral_helium_high_level_photoionization_cross_sections (28,F)` |
| light elements | `nitrogen_edge_cross_sections (3,F)`; `oxygen_911_cross_section (F,)`; `calcium_edge_cross_sections (3,F)`; `magnesium_ionized_cross_section_rows (14,F)` |
| C I | `carbon_boundfree_cross_section_rows (25,F)`; `carbon_freefree_prefactor (F,)`; `carbon_freefree_threshold (F,)` |
| Mg I | `magnesium_boundfree_cross_section_rows (15,F)`; `magnesium_freefree_prefactor (F,)`; `magnesium_freefree_threshold (F,)` |
| Si I | `silicon_boundfree_cross_section_rows (33,F)`; `silicon_freefree_prefactor (F,)`; `silicon_freefree_threshold (F,)` |
| Al I and Fe I | `aluminum_boundfree_cross_section (F,)`; `iron_boundfree_cross_section_rows (48,F)` |
| device views | `_tensor_cache`, keyed by `(field name, dtype string, device string)` |

These fields belong to the sampled extension object only. They must not be
added to `WindowInvariants` or a public structured-atmosphere archive merely
to make the chapter's architecture look more uniform.

## 5. Population ownership

The distinction between actual ion density and partition-normalized population
is not optional bookkeeping. The following mapping is exact.

| Process family | Population read | Meaning |
| --- | --- | --- |
| H- bound-free | H I stage-0 partition-normalized population and actual `electron_density`; H- itself is reconstructed by the 0.754209 eV Saha factor | implicit H- density |
| H- free-free | H I stage-0 partition-normalized population and actual `electron_density` | fitted free-free coefficient already carries its convention |
| H I bound-free | H I stage-0 partition-normalized population | level sums multiply statistical and Boltzmann factors explicitly |
| H II free-free | `hydrogen_ionized_population` when supplied; otherwise `hydrogen_partition_normalized_ion_stage_populations[:,1]` | standard pipeline's trimmed continuum view omits the optional actual-H-II field and takes the fallback |
| H2+ absorption | H I and H II partition-normalized stage columns | no stored H2 population is read |
| He- absorption | actual `helium_neutral_population` and actual `electron_density` | fitted continuum |
| He I bound-free | `partition_normalized_populations[:,0,1]` | neutral-helium stage divided by its partition function |
| He I free-free | actual `helium_singly_ionized_population` and actual electrons | He+ is the Coulomb partner |
| He II bound-free | `partition_normalized_populations[:,1,1]` | singly ionized helium level population base |
| He II free-free | `partition_normalized_populations[:,2,1]` and actual electrons | bare He stage; its partition convention is retained exactly |
| C I/Mg I/Al I/Si I/Fe I bound-free | named neutral partition-normalized population | explicit level/statistical-weight sums |
| Si II Peach | `partition_normalized_populations[:,1,13]` | temperature-bracketed table |
| sampled-extension N I/O I/Mg II/Ca II | corresponding partition-normalized stage population | present only in the precomputed sampled lane |
| hot-metal bound-free | 21 partition-normalized slots: C stages 0:4, N 0:5, O 0:6, Ne 0:6 | the 60-row transition table selects one slot per row |
| hot-metal Coulomb free-free | actual `ion_stage_populations`; `sum(Z_ion^2 n_ion)` over C,N,O,Ne,Mg,Si,S,Fe and charges 1...5 | never substitute partition-normalized values |
| H I Rayleigh | actual neutral H divided by a recomputed six-level H I partition, then multiplied by the ground statistical weight 2 | scattering population |
| He I Rayleigh | actual `helium_neutral_population` | scattering population |
| H2 Rayleigh in synthesis | H2 density reconstructed from actual neutral H, the six-level H partition, and an analytic equilibrium fit | stored `molecular_hydrogen_population` is not read |
| Thomson | actual `electron_density` | wavelength-independent at fixed `n_e/rho` |

`build_pops(...)` returns device tensors plus host float64 temperature arrays.
It builds:

- `hot_metal_populations (D,21)` from partition-normalized populations;
- `charge_square_population_sum (D,5)` from actual ion-stage populations;
- `_temperature_host` and `_natural_log_temperature_host`, both `(D,)`
  float64, for Si II bracket decisions.

The pipeline constructs a trimmed `_continuum_atmosphere` containing 18 exact
fields. It deliberately omits both `hydrogen_ionized_population` and
`molecular_hydrogen_population`. Direct calls to `continuum(...)` with a richer
mapping can therefore select the optional actual-H-II field, while the standard
pipeline takes the H-II fallback. Fixtures and oracles must exercise the
pipeline view, not only a full archive passed directly.

## 6. Complete synthesis process and routine matrix

“Standard” below means the pinned `SynthesisPipeline.run()` edge-triplet route.
“Extension” means implemented only through the explicitly supplied
`FrequencyInvariants` sampled route.

| Physical term | Abs./scat. | Exact routine(s) | Frequency/table law | Population and final scaling | Ownership |
| --- | --- | --- | --- | --- | --- |
| H- bound-free | absorption | `_hminus_bf_scalar`, `_hminus_opacity` | 85-point wavelength table; active for `nu > 1.82365e14`; parabolic remap; stored coefficient multiplied by `1e-18` | implicit H- factor, `s_nu`, `/rho` | standard |
| H- free-free | absorption | `_hminus_ff_table`, `_hminus_opacity` | 22 inverse-wavelength by 11 theta values; log wavelength then theta interpolation | H I normalized, `2 n_e`, `1e-26`, `/rho`; no extra external `s_nu` | standard |
| H I high-level tail | absorption | `_hydrogen_opacity` | analytic `nu^-3` tail between parity-pinned limits | H I normalized, thermal exponent difference, `s_nu`, `/rho` | standard |
| H I explicit bound-free | absorption | `_karzas_latter_cross_section`, `_hydrogen_opacity` | n=15...7, n=6...2, and n=1 threshold branches; Karzas-Latter tables | H I normalized; explicit weights and Boltzmann factors; `s_nu`; `/rho` | standard |
| H II Coulomb free-free | absorption | `_coulomb_freefree_gaunt`, `_hydrogen_opacity` | charge-1 Gaunt factor, `nu^-3 T^-1/2` | actual/fallback H II, `n_e`, `s_nu`, `/rho` | standard |
| H2+ | absorption | H2+ block in `_minor_terms` | analytic frequency and energy polynomials; active for `nu <= 3.28805e15` | H I normalized times `2 * H II normalized`, thermal exponential, `s_nu`, `/rho` | standard |
| He- | absorption | He- block in `_minor_terms` | analytic `T`, constant, and `1/T` fit with frequency-dependent coefficients | actual He I times `n_e`, three `1e15` scale divisions, `/rho`; no additional `s_nu` in source | standard |
| compact C I visible edge | absorption | compact branch in `_minor_terms` | one 22006.370 cm^-1 edge plus `1e-30` floor | C I normalized, Boltzmann weight, `s_nu`, `/rho` | standard |
| compact Mg I visible edges | absorption | compact branch in `_minor_terms` | five analytic edges plus `1e-30` floor | Mg I normalized, weights, Boltzmann, `s_nu`, `/rho` | standard |
| compact Al I visible edges | absorption | compact branch in `_minor_terms` | five analytic edges plus `1e-30` floor | Al I normalized and branch-specific stimulated factor, `/rho` | standard |
| compact Si I visible edge | absorption | compact branch in `_minor_terms` | one 17777.641 cm^-1 edge plus `1e-30` floor | Si I normalized, Boltzmann, `s_nu`, `/rho` | standard |
| full C I | absorption | `_carbon_neutral_boundfree_opacity`, `_metal_boundfree_opacity_grid` | 25 levels, two fine-structure limits, resonances, Kramers tail | C I normalized, `s_nu`, `/rho` | sampled fallback with layout `True`; grid extension |
| full Mg I | absorption | `_magnesium_neutral_boundfree_opacity`, `_metal_boundfree_opacity_grid` | 15 levels, analytic/Karzas-Latter groups, Kramers tail | Mg I normalized, `s_nu`, `/rho` | sampled fallback with layout `True`; grid extension |
| full Si I | absorption | `_silicon_neutral_boundfree_opacity`, `_metal_boundfree_opacity_grid` | 33 levels, fine-structure/resonance groups, Kramers tail | Si I normalized, `s_nu`, `/rho` | sampled fallback with layout `True`; grid extension |
| full Al I | absorption | `_aluminum_neutral_boundfree_opacity`, `_metal_boundfree_opacity_grid` | twin fine-structure edges | Al I normalized, `s_nu`, `/rho` | sampled fallback with layout `True`; grid extension |
| Fe I resonance forest | absorption | `_iron_neutral_boundfree_opacity`, `_metal_boundfree_opacity_grid` | 48 Lorentzian-quartic transition rows | Fe I normalized, Boltzmann, `s_nu`, `/rho` | sampled fallback with layout `True`; grid extension |
| He I bound-free and high-level tail | absorption | `_helium_opacity`; grid alternative `_helium_opacity_grid` | high-level tail, n=5...2 autoionization/analytic branches, excited thresholds; grid lane uses precomputed low/high photoionization arrays | He I normalized, explicit weights/Boltzmann, `s_nu`, `/rho` | standard scalar; alternate grid implementation in extension |
| He I free-free | absorption | `_helium_opacity` | charge-1 Coulomb Gaunt; standard scalar constant is `3.619e8`, layout-True scalar uses `3.6919e8` | actual He II, `n_e`, `s_nu`, `/rho` | standard, with lane-sensitive constant |
| He II bound-free | absorption | `_helium_opacity`, `_helium_opacity_grid` | hydrogenic high-level tail, n=9...1 transitions | He II normalized, explicit weights/Boltzmann, `s_nu`, `/rho` | standard and extension |
| He II free-free | absorption | `_helium_opacity`, `_helium_opacity_grid` | charge-2 Gaunt and `4 * 3.6919e8` | He III normalized, `n_e`, `s_nu`, `/rho` | standard and extension |
| hot-metal Coulomb free-free | absorption | `_hot_metal_and_silicon_singly_ionized_opacity`; grid `_hot_metal_opacity_grid` | charges 1...5, Gaunt table, `nu^-3 T^-1/2` | actual charge-square population sum, `n_e`, `s_nu`, `/rho` | standard and extension |
| hot-metal bound-free table | absorption | same hot-metal routines | 60 rows: threshold, cross-section, shape, power, multiplier, excitation, population slot | 21 normalized population slots; source retains a transition only where it exceeds the running hot opacity by 1 percent before adding its Boltzmann-weighted value | standard and extension |
| Si II Peach | absorption | `_silicon_singly_ionized_peach_opacity`; grid `_silicon_singly_ionized_peach_base_grid` | 14x6 log table, seven frequency thresholds, host-float64 temperature brackets | Si II normalized, `s_nu`, `/rho` | standard and extension |
| N I | absorption | `_light_element_frequency_grids`, `_light_element_opacity_grid` | three Seaton edges | N I normalized, weights/Boltzmann, `s_nu`, `/rho` | precomputed extension only |
| O I | absorption | same | 911-Angstrom Seaton edge | O I normalized, factor 9, `s_nu`, `/rho` | precomputed extension only |
| Mg II | absorption | same | 14 bound-free rows plus high-level tail | Mg II normalized, weights/Boltzmann, `s_nu`, `/rho` | precomputed extension only |
| Ca II | absorption | same | three analytic/Seaton edges | Ca II normalized, weights/Boltzmann, `s_nu`, `/rho` | precomputed extension only |
| H I Rayleigh | scattering | `_rayleigh_polarizability_factor`, `_scattering_opacity`; grid versions | Gavrila piecewise tables/fits | actual H I divided by six-level partition, ground weight 2, `/rho` | standard and extension |
| Thomson electron | scattering | `_scattering_opacity`, `_scattering_opacity_grid` | constant `0.6653e-24` | `n_e/rho`; frequency independent | standard and extension |
| He I Rayleigh | scattering | `_minor_terms`, `_minor_terms_grid` | analytic polarizability fit, frequency capped at `5.15e15` | actual He I `/rho` | standard and extension |
| H2 Rayleigh | scattering | `_minor_terms`, `_minor_terms_grid` | analytic cross-section, frequency capped at `2.922e15` | analytically reconstructed H2 `/rho`; does not read stored H2 | standard and extension |

The ordered standard sampled-column sum is:

1. H-;
2. H I;
3. minor absorption (H2+, He-, compact neutral metals);
4. He I;
5. He II;
6. hot-metal continuum;
7. Si II Peach.

Scattering is H I Rayleigh plus Thomson plus the minor He I and H2 Rayleigh
terms. Preserve this order in exact parity code because roundoff and the
hot-metal 1-percent acceptance condition make arbitrary reassociation unsafe.

## 7. Static data matrix

`continuum_tables.npz` contains exactly 25 float64 arrays:

| Group | Exact field and shape | Consumption |
| --- | --- | --- |
| Karzas-Latter | `karzas_latter_log10_frequency_hz (29,15)`; `karzas_latter_total_log10_cross_section_cm2 (29,15)`; `karzas_latter_angular_log10_cross_section_cm2 (6,6,29)`; `karzas_latter_high_level_energy_offset_rydberg (29,)` | H, He, C/Mg/Si/Mg II hydrogenic bound-free |
| H- bound-free | `hminus_boundfree_wavelength_nm (85,)`; `hminus_boundfree_cross_section_cm2 (85,)` | parabolic wavelength remap; numerical values are multiplied by `1e-18` by both atmosphere and synthesis code |
| H- free-free | `hminus_freefree_inverse_wavelength_grid (22,)`; `hminus_freefree_theta_grid (11,)`; `hminus_freefree_short_wavelength_table (11,11)`; `hminus_freefree_long_wavelength_table (11,11)` | log-wavelength and theta interpolation |
| H I Rayleigh | `hydrogen_rayleigh_gavrila_main_table (74,)`; `hydrogen_rayleigh_gavrila_ab_table (27,)`; `hydrogen_rayleigh_gavrila_bc_table (24,)`; `hydrogen_rayleigh_gavrila_cd_table (22,)`; `hydrogen_rayleigh_gavrila_lyman_continuum_table (64,)`; `hydrogen_rayleigh_gavrila_lyman_frequency_ratio_grid (64,)` | piecewise polarizability factor |
| Coulomb free-free | `coulomb_freefree_charge_log_offset (6,)`; `coulomb_freefree_gaunt_table (12,11)` | charges 1...6 available; continuum assembly uses 1...5 |
| hot metals | `hot_metal_boundfree_transition_table (60,7)` | threshold/shape/population-slot rows |
| Si II Peach | `silicon_singly_ionized_peach_cross_section_table (14,6)`; `silicon_singly_ionized_peach_threshold_frequencies_hz (7,)`; `silicon_singly_ionized_peach_natural_log_frequency_grid (9,)`; `silicon_singly_ionized_peach_natural_log_temperature_grid (6,)` | frequency and temperature brackets |
| H partition | `hydrogen_neutral_level_energy_cm (6,)`; `hydrogen_neutral_level_statistical_weight (6,)` | six-level partition for Rayleigh/H2 reconstruction |

`ContinuumTables.from_npz(...)` casts every stored field to host float64.
`ContinuumTables.from_dict(...)` derives H-level energies in eV and the H-
free-free log tables, then uploads only
`coulomb_freefree_gaunt_table_device` eagerly. Other frequency-only arrays stay
on the host until a helper uses or caches them.

`continuum_edge_grid.npz` contains:

| Field | Shape / dtype | Exact role |
| --- | --- | --- |
| `signed_continuum_edge_frequency_hz` | `(341,) float64` | edge magnitude; sign is retained in schema but `build_edge_sample_frequencies` takes `abs` |
| `continuum_edge_wavelength_nm` | `(341,) float64` | strictly increasing from 1 to 500,000 nm |
| `continuum_edge_wavenumber_cm` | `(341,) float64` | provenance/diagnostic field, not placed in the schema-v4 runtime view |
| `continuum_edge_sample_frequency_hz` | `(1020,) float64` | packaged precomputation; pipeline loader reads it, but the schema omits it and `continuum(...)` recomputes identical samples |
| `continuum_edge_midpoint_wavelength_nm` | `(340,) float64` | interpolation center |
| `continuum_edge_interval_width_squared_over_two_nm2` | `(340,) float64` | `(lambda_R-lambda_L)^2/2` |
| `__meta__` | `(236,) uint8` | JSON provenance for source `continua.dat` |

The packaged 1,020-sample vector is bitwise equal to the current
`build_edge_sample_frequencies(...)` formula. This equality must become a data
gate before the chapter teaches recomputation.

Atmosphere-only CH/OH/CIA requires two additional source products, staged under
an explicitly atmosphere-owned role:

- `continuum_opacity_tables.npz`: CH `(106,15)`, OH `(130,15)`, their two
  41-point partition tables, H2-H2 `(81,7)`, and H2-He `(81,7)` collision
  tables, in addition to atmosphere continuum arrays;
- `molecular_equilibrium_tables.npz`: `h2_partition_function (200,)`.

These atmosphere tables must not be silently merged into the synthesis
`ContinuumTables` interface.

## 8. Edge-triplet sampling and log-parabolic interpolation

For an interval with increasing wavelength nodes
`lambda_L < lambda_R` and midpoint `lambda_M`, the source evaluates:

\[
\nu_L = \frac{|\nu(\lambda_L)|}{1.0000001},\qquad
\nu_M = \frac{c}{(\lambda_L+\lambda_R)/2},\qquad
\nu_R = |\nu(\lambda_R)|\,1.0000001.
\]

The tiny offsets put the left sample just to the red side of the left edge and
the right sample just to the blue side of the right edge. At an exact internal
edge, `searchsorted(..., side="right") - 1` assigns the wavelength to the
interval on its red side, matching the first sample of that interval. The
chapter should visualize this one-sided choice rather than describe the three
points as generic evenly spaced samples.

For each requested wavelength, the source finds and clips its edge interval.
It evaluates only `3 * len(unique(edge_indices))` physical sample columns. The
340-interval table bounds this cost independently of the number of synthesis
pixels.

With

\[
d_2=\frac{(\lambda_R-\lambda_L)^2}{2},
\]

the exact basis functions are

\[
\begin{aligned}
L_L(\lambda)&=\frac{(\lambda-\lambda_M)(\lambda-\lambda_R)}{d_2},\\
L_M(\lambda)&=\frac{2(\lambda_L-\lambda)(\lambda-\lambda_R)}{d_2},\\
L_R(\lambda)&=\frac{(\lambda-\lambda_L)(\lambda-\lambda_M)}{d_2}.
\end{aligned}
\]

They sum to one and reproduce the three node values. The implementation first
floors each sampled absorption and scattering value at `1e-30`, interpolates
`log10(opacity)`, then exponentiates. Thus an inactive process becomes a small
positive interpolation floor in the final slabs. Component-level physical
zeros and final-slab interpolation floors must be tested separately.

Edge selection and basis construction are host float64. Basis vectors and
wavelength indices are transferred as small tensors to the continuum device.
The physical slabs never need to cross back to the host at this stage.

## 9. Device, dtype, and host-boundary contract

The public synthesis policy from `resolve_runtime(...)` is:

- CUDA if available, then MPS, then CPU;
- float64 on CUDA and CPU when dtype is omitted;
- float32 on MPS;
- an explicit float64 request on MPS raises `ValueError`.

The standard pipeline passes this resolved dtype to `ContinuumTables.from_npz`.
There is, however, an important standalone difference:
`ContinuumTables.from_npz(device="cpu", dtype=None)` falls through to module
`DEFAULT_DTYPE`, which is float32. It does not call `resolve_runtime`.
Reader code must either use the pipeline-resolved pair or pass the intended
dtype explicitly.

Actual placement:

- source tables, threshold searches, edge lookup, unique interval selection,
  H- wavelength/theta brackets, Si II frequency/temperature brackets, and
  several level cross-section constructions are host float64;
- populations and opacity columns are Torch tensors in work dtype on the
  selected device;
- the Coulomb Gaunt table has an eager device copy;
- `FrequencyInvariants.tensor(...)` lazily caches contiguous float64 host
  arrays converted to a requested `(dtype, device)` pair;
- the standard edge-triplet lane batches Coulomb-Gaunt evaluation over sample
  frequencies, then loops over at most 1,020 sample columns for the remaining
  scalar/table-sensitive work;
- continuum slabs remain work dtype. The all-backend float32 line accumulator
  introduced in Chapter 7 must not be retroactively applied here.

The host decisions are parity-sensitive. A GPU explanation must distinguish
large tensor arithmetic from discrete bracket and branch ownership; “GPU
synthesis” does not imply that all table logic is moved to the GPU.

## 10. Atmosphere-only molecular-continuum boundary

The atmosphere composite calls
`compute_molecular_continuum_opacity_columns(...)` when opacity flag 8 is on.
For cool layers it adds:

1. CH cross-section times CH partition, scaled by actual `ch_population`,
   stimulated emission, and `1/rho`;
2. OH cross-section times OH partition, scaled by actual `oh_population`,
   stimulated emission, and `1/rho`;
3. H2 collision-induced absorption at wavenumber `<= 20,000 cm^-1`, with
   H2-H2 and H2-He collision tables. Its H2 population is reconstructed from
   the hydrogen ground population, departure coefficient, and atmosphere H2
   partition table, and is forced to zero above 20,000 K.

The exact molecular-continuum table walks are:

| Branch | Frequency gate | Temperature gate/interpolation | Population scaling |
| --- | --- | --- | --- |
| CH | photon-energy index `20 <= floor(10 E_eV) < 105` | cross-section table on 0.1-eV rows and 500-K columns; 200-K partition table; zero for `T >= 9000 K` | packed-slot `ch_population / rho`, partition factor, stimulated emission |
| OH | shifted photon-energy index `0 < floor(10 E_eV)-20 < 130` | same two-stage energy/temperature interpolation pattern; zero for `T >= 9000 K` | packed-slot `oh_population / rho`, partition factor, stimulated emission |
| H2-H2/H2-He CIA | wavenumber `<= 20,000 cm^-1`, 250-cm^-1 rows | temperature index clipped to 1...6 over the 1,000-K columns; retain the source's exact two-column weighting order | reconstructed H2 times `(H2 * sigma_H2H2 + He I * sigma_H2He) / rho`, stimulated emission |

`build_continuum_atmosphere_state(...)` gets CH and OH from exact packed
partition-normalized slots 845 and 847. It also carries hydrogen departure
coefficients. Atmosphere H2+ multiplies its ground/ion population product by
the hydrogen ground departure coefficient; synthesis H2+ uses the analogous
polynomial but has no departure-coefficient field. “Present in both” therefore
does not mean the two backend state conventions are interchangeable.

The atmosphere scattering composite separately includes H2 Rayleigh when flag
12 is active, with its atmosphere H2 population convention. The standard
all-flags path also includes H I, He I, and electron scattering.

The synthesis composite contains no CH or OH population key and never calls an
H2-CIA table. It does include:

- H2+ absorption in `_minor_terms`;
- H2 Rayleigh in `_minor_terms`, using a different analytic reconstruction of
  H2 from neutral hydrogen rather than the schema's stored H2 column.

The chapter should use one cool-state feature matrix:

| Process | Atmosphere composite | Standard synthesis composite |
| --- | --- | --- |
| CH continuum | yes | no |
| OH continuum | yes | no |
| H2-H2 / H2-He CIA | yes | no |
| H2+ absorption | yes | yes |
| H2 Rayleigh | yes | yes, with synthesis-specific reconstructed population |

This is a model boundary, not a bug to “repair” in textbook code.

## 11. Exact teaching sequence

The reader-facing chapter should keep the two movements in the existing global
contract, with the following dependency order.

### Movement 5A — turn populations into smooth absorption

1. **Opening observation.** Show a one-panel solar-like opacity budget after
   line records are removed. Ask why the background remains structured.
2. **One absorber, one photon.** Use a threshold schematic to distinguish
   bound-free from free-free absorption. Derive `n sigma / rho` and check the
   units immediately.
3. **Net rather than gross absorption.** Derive `s_nu`; plot its low- and
   high-`h nu/kT` limits in one compact panel.
4. **Build H-.** Derive the implicit H- population factor, then implement
   bound-free and free-free as separate bite-sized functions. Show the
   threshold and a solar diagnostic where H- is important.
5. **Build hydrogen.** Start with one bound level, add the explicit level
   ladder, the high-level tail, and H II free-free. Explain why the first uses
   partition-normalized H I and the last uses H II.
6. **Add H2+ and He-.** Give each process one physical paragraph and one
   limiting check; do not bury them inside a “minor” label in prose.
7. **Build helium.** Treat He I bound-free/free-free and He II
   bound-free/free-free in the same population-to-cross-section-to-mass-opacity
   pattern. Use hot versus solar columns to show regime activation.
8. **First ordered budget.** Sum only the standard H/He/light-particle
   absorption terms in source order. The output is a checked `(D,F)` sampled
   absorption budget, not yet the final continuum.

### Movement 5B — metals, redirection, and exact wavelength placement

9. **Why metals shape a continuum without metal lines.** Introduce compact C I,
   Mg I, Al I, and Si I standard branches, then hot-metal free-free/bound-free
   and Si II Peach. Keep the full neutral-metal and N/O/Mg II/Ca II
   `FrequencyInvariants` versions in a clearly labeled implemented-extension
   subsection.
10. **Actual versus normalized populations.** Perturb one view at a time:
    bound-free must follow the normalized view; charge-square free-free must
    follow actual ion-stage populations.
11. **Scattering is redirection.** Derive Thomson scaling and contrast it with
    blue-rising H I, He I, and H2 Rayleigh scattering. Keep the absorption and
    scattering slabs separate.
12. **State the molecular boundary.** Use the five-row feature matrix above.
    Explain CH/OH/CIA physically once, show the atmosphere function that owns
    them, and prove their absence from the synthesis call trace. Do not add
    them to synthesis.
13. **Why three samples per edge.** Draw two adjacent thresholds and the
    one-sided triplet. Predict what happens exactly at the shared edge.
14. **Derive the three basis functions.** Verify node reproduction and
    partition of unity before applying them to opacity.
15. **Interpolate logarithmic opacity.** Explain the `1e-30` floor, evaluate
    only used intervals, and build complete `(D,W)` absorption and scattering
    slabs.
16. **Destination check.** Run the reader-built standard route first, then load
    pinned component and final-slab goldens. Close with the exact outputs and
    the unresolved narrow-line forest that causes Chapter 6.

Use three original schematics, not source screenshots:

- population -> cross-section -> mass opacity;
- absorption versus scattering;
- the one-sided edge triplet and log-parabolic reconstruction.

Every plotted budget should be one panel, one selected depth or a small number
of directly compared depths, and a restrained set of named components. There
is no detached exercise section; all limiting and perturbation checks belong
where they answer the live question.

## 12. Fixture, oracle, and parity requirements

### 12.1 Input products to stage

Stage only provenance-bound copies:

1. synthesis `continuum_tables.npz`;
2. synthesis `continuum_edge_grid.npz`;
3. a compact atmosphere-owned subset containing the exact CH/OH/H2-H2/H2-He
   arrays and H2 partition data, or the two full atmosphere tables if the
   subset transformation cannot be made independently auditable;
4. four schema-v4 atmospheres: `hot_dwarf`, `solar_dwarf`,
   `low_gravity_giant`, and `cool_molecule_rich`;
5. compact wavelength selections that include:
   - both sides of one H- or metal threshold;
   - an exact internal edge plus `nextafter` points on each side;
   - a red/infrared H2-CIA diagnostic wavelength;
   - a blue Rayleigh diagnostic wavelength;
   - UV points that activate the extension-only light-element terms.

Every staged array needs role, original path, pinned commit, SHA-256, dtype,
shape, axes, and unit in `data/MANIFEST.json`.

### 12.2 Golden products

Capture comparison-only goldens after the textbook calculation exists:

- `build_pops` outputs, including the 21 hot-metal slots and five
  charge-square columns;
- each named standard absorption component at selected sample frequencies;
- each named standard scattering component;
- ordered partial sums after every standard addition;
- exact used edge indices, sample indices, and sample frequencies;
- sampled standard absorption/scattering `(D,3U)`;
- final interpolated absorption/scattering `(D,W)`;
- the sampled diagnostic `continuum_source`;
- atmosphere-only CH, OH, H2-H2/H2-He CIA components at the cool fixture;
- extension-only full-metal/light-element component arrays, stored under an
  explicit nonstandard-route role.

Golden metadata must include the source commit, all input hashes, route name,
`coulomb_table_energy_first`, fixture hash, wavelength/frequency vector,
device, dtype, Torch/NumPy versions, and tolerance.

### 12.3 Mandatory analytic and dimensional gates

1. `n sigma / rho` reduces to cm^2 g^-1.
2. `s_nu` stays in `[0,1]` and has the correct asymptotic limits.
3. Every bound-free branch is zero on its physically inactive side before the
   final interpolation floor, except for the source's intentional `1e-30`
   compact metal floors.
4. Doubling an isolated absorber population doubles its contribution.
5. Doubling `rho` while holding number densities fixed halves the isolated
   mass opacity.
6. Thomson scattering is frequency independent at fixed `n_e/rho`.
7. Isolated Rayleigh terms rise strongly blueward over a selected valid
   interval.
8. Actual and normalized population perturbations activate only their owned
   terms.
9. All final sampled and interpolated slabs are finite and nonnegative over
   the chapter's supported window.

### 12.4 Edge and interpolation gates

1. The staged 341-edge arrays have the exact pinned hashes and shapes.
2. `build_edge_sample_frequencies(...)` is bitwise equal to the packaged
   1,020-sample field.
3. Every in-range requested wavelength has one interval; exact internal edges
   choose the red-side interval.
4. Evaluation count is exactly `3 * number_of_unique_used_intervals`.
5. No unused interval invokes a physical opacity evaluation.
6. All three basis functions sum to one.
7. They reproduce the stored triplet log opacities at left, midpoint, and
   right nodes.
8. The interpolated result is equal to `10**(basis-weighted log10 samples)`.
9. A sign flip of `signed_continuum_edge_frequency_hz` leaves the current
   runtime result unchanged because the sampler uses magnitudes; preserve the
   signed stored field anyway.
10. Direct `continuum(...)` and the staged textbook standard driver match
    component-wise and in the final `(D,W)` slabs.

### 12.5 Route-ownership gates

1. A call trace from `SynthesisPipeline.run()` reaches `continuum(...)` with no
   `pops` and no `FrequencyInvariants`.
2. `WindowInvariants` contains `continuum_tables` but no
   `FrequencyInvariants`.
3. The standard route uses `coulomb_table_energy_first=False`.
4. `compute_sampled_continuum(...)` uses `True`.
5. The explicitly precomputed route is tested independently and never reported
   as the standard pipeline result.
6. CH/OH/CIA functions are called by the atmosphere composite and never by the
   synthesis composite.
7. H2+ and H2 Rayleigh are nonzero in a deliberately selected synthesis state.
8. Multiplying the schema `molecular_hydrogen_population` by a large factor
   leaves the standard synthesis continuum unchanged; perturbing neutral H
   changes the reconstructed H2 Rayleigh term.
9. Passing a richer atmosphere mapping directly and passing the pipeline's
   trimmed view are compared explicitly for the optional H-II fallback.

### 12.6 Four-regime and backend gates

For every regime, store both active and physically justified near-zero
branches:

- `hot_dwarf`: weak H-, strong ion/electron and helium tests;
- `solar_dwarf`: H- diagnostic and stable standard continuum;
- `low_gravity_giant`: density scaling and scattering fraction;
- `cool_molecule_rich`: H2+, H2 Rayleigh, and atmosphere-only CH/OH/CIA.

Run CPU float64 as the reference. Compare CUDA float64 when available and MPS
float32 when available under separately recorded tolerances. Verify cold versus
warm `WindowInvariants` cache equality, noting that this cache owns
`ContinuumTables`, not `FrequencyInvariants`. Verify standalone
`ContinuumTables.from_npz(..., dtype=None)` returns float32 and do not confuse
that behavior with the public pipeline's resolved CPU/CUDA float64 default.

## 13. P0 ambiguities that must be resolved before implementation

These are source/design contradictions, not optional polish.

### P0.1 — “Complete continuum” currently names different process sets

The standard pipeline's `continuum(...)` uses compact neutral-metal branches,
the default Coulomb layout, and no explicit light-element grid.
`compute_sampled_continuum(...)` forces the alternate layout. Supplying
`FrequencyInvariants` also activates N I/O I/Mg II/Ca II and different
materialized He/metal helpers.

**Required resolution:** freeze the standard lane as the chapter's destination
and label the other lane as an implemented extension. Do not publish one
blended component list as if all terms run in `SynthesisPipeline`.

### P0.2 — full sampled invariants are not universally scalar-equivalent

A read-only diagnostic using the pinned
`examples/data/sun_structured_atmosphere.npz` showed:

- the invariant and non-invariant sampled lanes agree closely at ordinary
  optical points;
- across 20--5000 nm, the invariant lane produced non-finite absorption below
  roughly 33 nm for this fixture;
- among finite points the largest observed relative absorption difference was
  about 5.6 percent near 111.7 nm;
- the standard layout-false sampled values also differ from the layout-true
  sampled values (for example at 300 nm in a middle solar layer).

This is consistent with the route-specific source branches and disproves a
global “performance-only, scalar/batched bit parity” assumption.

**Required resolution:** choose and record the chapter's supported wavelength
domain. Test common helpers component-wise; do not require whole-slab
scalar/invariant parity outside a domain where it has actually been
established.

### P0.3 — the stored H2 population is not synthesis-continuum input

The schema and global brief imply that `molecular_hydrogen_population` is read
by continuum. The pinned synthesis `build_pops(...)` does not read it, and the
pipeline's continuum view omits it. H2 Rayleigh reconstructs its own population
from neutral H.

**Required resolution:** correct the Chapter 5 reads contract. Retain the
stored H2 field for hydrogen-line collision state and other later consumers,
but do not claim it controls synthesis continuum.

### P0.4 — H- table field unit is semantically misleading

The field is named `hminus_boundfree_cross_section_cm2`, but both atmosphere
and synthesis multiply its tabulated values (0--95) by `1e-18` before treating
them as cm^2.

**Required resolution:** the manifest must describe the stored numeric
convention separately from the physical post-conversion unit. Do not silently
rename the exact field and do not label the raw numbers as already being cm^2.

### P0.5 — omitted dtype has two meanings

The public pipeline resolves CPU/CUDA omitted dtype to float64.
`ContinuumTables.from_npz(device="cpu", dtype=None)` uses module
`DEFAULT_DTYPE=float32`.

**Required resolution:** progressive textbook code must pass the resolved
dtype explicitly. A visible check should demonstrate the distinction once.

### P0.6 — optional H-II ownership depends on the mapping view

`build_pops(...)` prefers an optional actual `hydrogen_ionized_population`.
The standard pipeline trims that field, so it uses the second
partition-normalized H-stage column. A direct call with the full structured
builder output can take the other branch.

**Required resolution:** standard-route oracles must pass the exact pipeline
view. Add an equality/near-equality check for the two H-II columns in all
fixtures and document the fallback without inventing a new public field.

### P0.7 — signed edge semantics are stored but erased by current sampling

The schema says the sign carries an interpolation-side convention, while the
runtime sampler takes `abs` at both ends.

**Required resolution:** preserve and hash the signed source field, teach that
the current sampler uses its magnitude, and add a sign-invariance test. Do not
invent an operational sign effect absent from the pinned source.

## 14. No-redundancy and handoff contract

Chapter 5 may recap only:

- Chapter 1's meaning of absorption, scattering, and `B_nu`;
- Chapter 3's actual versus partition-normalized ion populations;
- Chapter 4's existence of molecular state.

It must not rederive Saha balance, partition functions, or molecular
equilibrium. It must not introduce oscillator strengths, Voigt profiles, line
wings, Rosseland means, or transfer iteration. Atmosphere CH/OH/CIA appears
once as a model-boundary comparison, not as a second atmosphere-opacity
chapter.

The chapter closes only when it has produced and parity-checked:

- `continuum_absorption (D,W)` in cm^2 g^-1;
- `continuum_scattering (D,W)` in cm^2 g^-1;
- a named component budget that sums in the exact standard order;
- a route claim table separating standard, sampled diagnostic, sampled
  extension, and atmosphere-only terms.

The causal bridge is then simple: these smooth slabs explain the broad
background and set the local line-selection scale, but they cannot create the
narrow absorption forest. Chapter 6 must therefore build one bound-bound line
from its level population, oscillator strength, Doppler width, damping, and
normalized profile.

## 15. Acceptance checklist for the Chapter 5 wave

- [ ] All P0 decisions above are reflected in
  `design/part3_part4_synthesis_brief.md` and the global reads/writes contract.
- [ ] Staged tables and fixtures have exact provenance and manifest hashes.
- [ ] Standard, sampled diagnostic, extension, and atmosphere-only oracles
  carry distinct roles.
- [ ] Component goldens precede total-slab goldens.
- [ ] The standard pipeline's trimmed atmosphere view is used in parity tests.
- [ ] Edge-side, exact-edge, and unused-interval behavior is tested.
- [ ] CH/OH/CIA absence and H2+/H2-Rayleigh presence are executable claims.
- [ ] CPU/CUDA/MPS dtype policies and the standalone dtype exception are tested.
- [ ] Every notebook code cell has one conceptual purpose and no source dump.
- [ ] Every plot has one physical claim, professional typography, and readable
  units.
- [ ] The chapter ends with a summary and a causal link to Chapter 6.
