# Chapter 6 exact source contract — One Spectral Line

Status: pre-implementation source and parity contract
Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`
Audited on: 2026-07-30

This document freezes the exact ordinary bound-bound line physics and source
routes that Chapter 6 may teach and implement. It does **not** yet freeze the Fe
I teaching record, atmosphere integration fixture, or comparison goldens; the
gates in Sections 15–17 own those decisions. The contract is deliberately
narrower than a line-opacity module inventory. Chapter 6 constructs one
ordinary line from one readable record, first at one depth and then through all
supplied depths. Chapter 7 owns catalog decoding, line selection, window
margins, sparse forests, concurrent deposition, and every hydrogen, helium,
autoionizing, COR, PRD-routing, and merged-series branch. Chapter 8 owns
molecular bands.

The read-only source and paper trees were not modified.

## 1. Reader destination

The reader should leave Chapter 6 able to explain and compute

\[
\begin{aligned}
\kappa_{\nu,l}={}&
\frac{\pi e^2}{m_e c}\,
\frac{n_{s,r}}{\rho U_{s,r}}\,(gf)_l
\exp\left(-\frac{E_l}{k_{\rm B}T}\right) \\
&\times
\left[1-\exp\left(-\frac{h\nu}{k_{\rm B}T}\right)\right]
\phi_l(\nu),
\qquad
\int_{-\infty}^{\infty}\phi_l(\nu)\,{\rm d}\nu=1 .
\end{aligned}
\]

This equation assumes LTE level populations and stimulated emission for one
ordinary, isolated bound-bound transition. Blends, magnetic splitting, and the
special-profile routes listed in Section 2.4 are outside this one-line
construction. Here \(e\) is the magnitude of the electron charge in CGS
electrostatic units, \(m_e\) is the electron mass in g, \(c\) is the speed of
light in cm s\(^{-1}\), \(h\) is Planck's constant in erg s, and \(k_{\rm B}\)
is Boltzmann's constant in erg K\(^{-1}\).

Every factor has one job:

- `partition_normalized_populations` supplies \(n_{s,r}/U_{s,r}\), not a
  bound-level population;
- \((gf)_l\) includes the lower-level statistical weight;
- `lower_excitation_cm` and `hc_over_kt` form the Boltzmann exponent;
- `fractional_doppler_widths` supplies
  \(\delta_l=\Delta v_{\rm D}/c=\Delta\nu_{\rm D}/\nu_l\);
- `electron_density` and `collision_density_proxy` turn the three damping
  coefficients into a total Lorentz width;
- the Chapter 5 stimulated-emission factor converts the gross absorption into
  net LTE line absorption;
- the normalized profile redistributes a fixed integrated strength without
  creating absorber population.

The exact reader-facing product is one
`line_mass_absorption_coefficient` array with axes `(depth, wavelength)`,
unit cm\(^2\) g\(^{-1}\), together with named one-depth intermediates. It is
not a flux, a normalized spectrum, a catalog, or a line forest.

The native array coordinate is wavelength, but the deposited value is not a
new opacity density per unit wavelength. At wavelength sample \(\lambda_i\),
define

\[
\kappa_{l,i}\equiv\kappa_{\nu,l}\!\left(\nu=\frac{c}{\lambda_i}\right).
\]

No \(|{\rm d}\nu/{\rm d}\lambda|\) Jacobian is applied when the extinction
coefficient is evaluated on that grid. The profile-area statement remains an
integral over frequency, or equivalently over dimensionless Doppler offset
\(u\); a raw sum over nm samples is not a normalization test.

## 2. Four boundaries that must remain visible

### 2.1 Physical profile versus deposited grid values

The mathematical profile \(\phi_l(\nu)\) is normalized in continuous
frequency. The production kernels instead evaluate a tabulated Harris
approximation at discrete grid points, apply branch-specific shortcuts, stop
at continuum-relative cutoffs, and deposit in `float32`.

Therefore:

- the analytic/dense teaching profile receives the normalization check;
- the deposited production slab receives exact source-parity and dtype checks;
- a truncated, cutoff-dependent discrete sum must not be advertised as
  exactly unit normalized.

### 2.2 Gross line opacity versus net LTE line opacity

The atmosphere line accumulator returns a pre-stimulated line slab. The
atmosphere transfer kernel later multiplies each frequency column by
`stimulated_all`.

The synthesis `accumulate_atomic(..., apply_stim=True)` route applies

```python
1.0 - torch.exp(-frequency_grid_hz * photon_temperature_factor)
```

after the ordinary atomic deposits and returns the net line slab.

The same physical factor is reused; its exact lifecycle differs. Cross-lane
comparisons must align this boundary before comparing values.

### 2.3 Ordinary profile physics versus catalog machinery

Chapter 6 may read one provenance-bound ordinary record and expose its physical
fields. It may not teach:

- fixed-width or packed catalog parsing;
- isotope and energy-shift correction families beyond the chosen record's
  already-declared zero-shift controlled case;
- default-damping generation for missing catalog values;
- line-size margins, sorting, cache keys, or source shards;
- atmosphere line-selection inequalities or fused keep kernels;
- many-line chunking, scatter-add race avoidance, or reduction ordering.

Those are Chapter 7 responsibilities. A one-record production call may be used
as a parity oracle after the one-line physics has been built.

### 2.4 Ordinary lines versus special profiles

Only ordinary `line_type == 0` is needed for the teaching record.
`line_type == 3` follows the same ordinary LTE metal-opacity kernel in the
pinned synthesis and atmosphere code, but its PRD label and routing semantics
belong to Chapter 7.

The following do not enter Chapter 6:

| deferred record/profile | exact destination |
| --- | --- |
| type `1` autoionizing Shore–Fano | Chapter 7 |
| type `2` COR, parsed but skipped/unwired | Chapter 7 |
| H I and D fine structure, Stark wings, dissolution, merging | Chapter 7 |
| He I, \(^{3}\)He, He II, merge tapers | Chapter 7 |
| merged-series continua and selectors | Chapter 7 |
| ordinary and special many-line selection/deposition | Chapter 7 |
| text-band, TiO, and H2O molecular records | Chapter 8 |

## 3. Pinned source identities

### 3.1 Atmosphere lane

| source | SHA-256 | Chapter 6 responsibility |
| --- | --- | --- |
| `payne_zero_atmosphere/constants.py` | `ac1f1fbd345dc816eb3e70a8f97ebebc7a4c744fd2759b32ec19f8c88d987036` | exact/reference constant tiers |
| `payne_zero_atmosphere/atmosphere_io.py` | `95c4d2cab230f6925e9404639ecb05b25af8c0c85755ac1ca70d760156a8683e` | `thermal_energy_erg` and `hc_over_kt` definitions |
| `payne_zero_atmosphere/doppler.py` | `e118a78bf5250ef5e1f77d652c9e78fbb7b92acf5c069f717faed7a3b3ea98f0` | packed fractional Doppler widths and strength support |
| `payne_zero_atmosphere/line_profile_math.py` | `9a5794140f00ff3c3fb6c2e3b28461bbc22b471f962d275055c066ad7f8acd15` | FASTEX and scalar Harris basis/evaluator |
| `payne_zero_atmosphere/line_catalog.py` | `2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92` | exact one-record container fields |
| `payne_zero_atmosphere/line_opacity.py` | `d0f9c43919be58a42547e12b7abc22161a7558bf17abbcd375ab04ccf57d7cc6` | compiled ordinary-line accumulation |
| `payne_zero_atmosphere/runner.py` | `05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7` | standard selected-line composition boundary |
| `payne_zero_atmosphere/transfer_kernels.py` | `50e759a085e6aefdb7819a3dbe3ef5e83405834f4b07e0a4de2f3c0e7354d3b9` | downstream stimulated-emission application |

`payne_zero_atmosphere/line_selection.py`, SHA-256
`b2c62fdf5e1fe43f33022184bfeff88985b13331354e3c745c7dab3a6b634fef`,
is an audited Chapter 7 boundary. Chapter 6 may receive one already-formed
`SelectedLineCatalog`; it may not reconstruct or summarize the bulk selector.

### 3.2 Synthesis lane

| source | SHA-256 | Chapter 6 responsibility |
| --- | --- | --- |
| `payne_zero_synthesis/constants.py` | `ed58004196790f9fb4a2871044c9cd36bf7bc42046a923f9314f7b8ea7456798` | exact line constants and source-catalog conversion fence |
| `payne_zero_synthesis/device.py` | `22e769ebed60ad3a0f2060264247e469a99afd20ec5cadb69a01b6e5fa82ea3c` | device, work dtype, and accumulation dtype |
| `payne_zero_synthesis/atomic_lines.py` | `0fa52833fb16487da1d5bfaaf5628a46751f888c1a57894a5037daa6d6667ab0` | readable one-record fields and normalized damping convention |
| `payne_zero_synthesis/line_opacity.py` | `639b95c3812f1a7d227b797fa89a4d6ef9725d5f0e1284f3d49cf86844278275` | Harris/FASTEX, invariant construction, and ordinary Torch deposit |
| `payne_zero_synthesis/pipeline.py` | `465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22` | fixed-state inputs and standard `apply_stim=True` composition |

The paper equation used to reconcile notation is in pinned
`main.tex`, SHA-256
`e11507b9150550b246f6664debf22e540aa92d8261eb40daabb594da91bd8e0d`.
The paper is explanatory evidence, not a reader runtime dependency.

## 4. Exact static data

### 4.1 Atmosphere Harris source

`source_data_files/atmosphere_tables/line_opacity_tables.npz`

- SHA-256:
  `89f486122cb8939b23dc5423145a46d88a77df8daf57a1def35055b7b8205f16`;
- byte size: `1,359`;
- `voigt_interpolation_table`: `(81,)`, NumPy `float64`;
- `hydrogen_profile_table`: `(81,)`, NumPy `float64`.

`load_line_opacity_tables()` insists on both keys. `build_voigt_profile_basis()`
remaps them to the 2001-point grid

```python
profile_grid = np.arange(2001, dtype=np.float64) / 200.0
```

and constructs:

- `gaussian_profile = exp(-u**2)`;
- `first_correction` from the exact `_remap_profile_table` ordering;
- `second_correction = gaussian_profile - 2*u**2*gaussian_profile`.

The result is a cached CPU NumPy `float64` `VoigtProfileBasis`.

The selected-line lane also constructs, rather than loads,

```python
build_selection_log_lookup()
```

as a `(32768,)` NumPy `float64` array

\[
\mathrm{TABLOG}[i-1]=10^{(i-16384)\,0.001},
\qquad i=1,\ldots,32768.
\]

`accumulate_selected_line_opacity` passes a contiguous `float32` cast of this
table into the compiled kernel. All five packed lookup indices are one-based
and must lie in 1…32768. The exact packed-wavelength reconstruction and
damping scales are

```python
_RATIO_LOG_STEP = log(1.0 + 1.0 / 2_000_000.0)
vacuum_wavelength_nm = exp(packed_wavelength_index * _RATIO_LOG_STEP)
_DAMPING_SCALE = 1.0 / 12.5664 / LIGHT_SPEED_NM_PER_S
```

The compiled path multiplies each TABLOG damping value by the reconstructed
wavelength and `_DAMPING_SCALE`, and performs those selected-line strength and
damping operations through the source's explicit `float32` casts. Chapter 6
must preserve those casts in parity tests without teaching the upstream
packing algorithm.

### 4.2 Synthesis Harris source

`source_data_files/synthesis_tables/line_profile_tables.npz`

- SHA-256:
  `87b47fc76bed10455218f43c4b6686525b961002e72d6a5ef01255a08deb27d4`;
- byte size: `57,308`;
- Chapter 6 consumes only
  `harris_profile_h0_table`, `harris_profile_h1_table`, and
  `harris_profile_h2_table`, each `(2001,)` NumPy `float64`;
- the other hydrogen/Stark arrays stay unopened until Chapter 7.

These arrays are uploaded by `precompute_invariants` to the synthesis device in
the lane's work dtype.

### 4.3 The two Harris bases are not one deduplicated table

The read-only audit reconstructed the atmosphere basis and compared it to the
synthesis archive:

- H0 and H2 differ at one extremely small tail entry by less than
  \(1.5\times10^{-44}\);
- H1 differs at indices 21–39, corresponding to \(u=0.105\ldots0.195\);
- the maximum H1 difference is
  `0.005044391583364227` at \(u=0.15\).

The files, loaders, and parity gates remain separate. “Both use Harris” is a
physical statement, not permission to replace either table.

### 4.4 Mass data already owned by Chapter 3

Chapter 6 reuses, without copying:

- atmosphere `major_isotope_mass_amu (1006,)` from
  `data/static/atmosphere_tables/isotope_tables.npz`, SHA-256
  `53c8d315fb53f1e051dc2752b028fc270d7c17a2c1042279c04ffcb750aef5c6`;
- synthesis `atomic_mass_amu (99,)` from
  `data/static/synthesis_tables/atomic_masses.npz`, SHA-256
  `d4739fef7e03964aea5a7b2604f9585fd9095c26c58f5b7d5d040aaafeb5d117`.

The packed atmosphere mass and synthesis element-mass policies remain
distinct. Chapter 6 uses the already-computed
`fractional_doppler_widths`; it does not reopen the layout construction.

## 5. One readable ordinary source record

The normal book must not copy the 246 MB synthesis source archive. The full
authority is:

`source_data_files/source_catalogs/lines/atomic_source_lines_parsed.npz`

- SHA-256:
  `4eafa927c02a4f74401523149a44e35239f2aaecb4a64f2905a4cd5530c2dde7`;
- byte size: `258,021,389`;
- 1,939,975 source rows.

The independently accepted Chapter 6 teaching-record candidate is zero-based
raw row `873702`. The acceptance report is
`design/chapter06_fe_record_candidate_audit.md`. It was chosen because it is
an ordinary Fe I record with:

- blank `line_category_tag`, hence ordinary `line_type == 0`;
- zero isotope-log corrections;
- blank energy-shift field and zero isotope wavelength shift;
- explicit, nonzero radiative, Stark, and van der Waals entries, so no default
  damping branch is needed;
- no hydrogen, helium, autoionizing, or molecular semantics.

The one-row teaching subset preserves the source archive's exact field names:
`stored_wavelength_nm`, `raw_log_oscillator_strength`, `species_code`,
`first_energy_column_cm`, `second_energy_column_cm`,
`radiative_damping_log`, `stark_damping_log`,
`van_der_waals_damping_log`, `lower_principal_quantum_number`,
`upper_principal_quantum_number`, `primary_isotope_number`,
`primary_isotope_log_correction`, `secondary_isotope_log_correction`,
`energy_shift_field`, `isotope_shift_units`, `line_size`, and
`line_category_tag`. The blank fixed-width `energy_shift_field` must remain a
byte string in the subset rather than being normalized into an invented
numeric value.

The audited row gives:

| physical/source field | exact audited value |
| --- | ---: |
| `species_code` | `26.0` |
| `stored_wavelength_nm` | `499.0341` nm |
| derived `wavelength_nm` | `499.03411946178176` nm |
| `lower_excitation_cm` | `33507.123` cm\(^{-1}\) |
| upper excitation | `53545.833` cm\(^{-1}\) |
| source \(\log_{10}(gf)\) | `-0.86` |
| `oscillator_strength` \(=gf\) | `0.1380384264602885` |
| raw radiative \(\gamma\) | `2.9512092266663903e8` s\(^{-1}\) |
| raw Stark coefficient | `3.5481338923357534e-5` cm\(^3\) s\(^{-1}\) |
| raw van der Waals coefficient | `3.090295432513592e-8` cm\(^3\) s\(^{-1}\) |
| `raw_radiative_damping_log` | `8.47` |
| `raw_stark_damping_log` | `-4.45` |
| `raw_van_der_waals_damping_log` | `-7.51` |
| derived `atomic_number`, `ion_stage`, `line_type` | `26`, `1`, `0` |

At this line's frequency, the exact `_build_records` normalization gives:

| exact production field | value |
| --- | ---: |
| `radiative_damping` | `3.909296919359072e-08` |
| `stark_damping` | `4.700008650504819e-21` cm\(^3\) |
| `van_der_waals_damping` | `4.093536407068315e-24` cm\(^3\) |
| `classical_line_strength` | `3.4403587659200408e-18` cm\(^2\) |

The subset must retain the source row index, source archive hash, all raw
fields needed to reproduce these values, the extraction command, shapes,
dtypes, units, and its own hash. The chapter derives this one record
transparently. Chapter 7 later generalizes the catalog transformations.

This record has passed the source, four-regime visibility, continuum-cutoff,
and synthesis loop-versus-batched checks. Its immutable raw teaching subset is
now published as
`data/subsets/chapter06_fe_i_source_row_873702.npz` (8,665 bytes, SHA-256
`bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04`).
The subset contains the 17 exact raw fields and provenance only—no derived
record, continuum, opacity, or golden output. Here, “visible in a regime” has
one exact meaning: the production center gate is active in at least one of the
six supplied depths. The measured active-depth counts are `3/6` for the hot
dwarf and `6/6` for each of the solar dwarf, low-gravity giant, and cool
molecule-rich regimes. Requiring every depth to pass would be a different and
physically unjustified rule; the real hot-dwarf cutoff rejects its three
deepest supplied layers. The physical teaching-record choice still remains a
candidate until both lane-specific fixture/oracle gates below close. If any
remaining gate fails, replacement must use the same explicit-selection
criteria and record its source row; silent hand tuning is forbidden.

## 6. Exact physical notation and production names

| mathematical object | physical meaning | exact name | unit / shape |
| --- | --- | --- | --- |
| \(\lambda_l\) | line rest wavelength | `wavelength_nm` | nm, scalar per line |
| \(\nu_l=c/\lambda_l\) | line rest frequency | local derived value | Hz |
| \(E_l\) | lower-level excitation | `lower_excitation_cm` | cm\(^{-1}\) |
| \((gf)_l\) | oscillator strength times lower statistical weight | `oscillator_strength` | dimensionless, scalar per line |
| \(n_{s,r}/U_{s,r}\) | partition-normalized ion-stage population | `partition_normalized_populations` | cm\(^{-3}\), `(D,6,139)` |
| \(\rho\) | mass density | `mass_density` | g cm\(^{-3}\), `(D,)` |
| \(hc/kT\) | converts wavenumber to Boltzmann exponent | `hc_over_kt` | cm, `(D,)` |
| \(\delta_l\) | fractional Doppler width \(\Delta v_D/c\) | `fractional_doppler_widths` | dimensionless, `(D,6,139)` |
| \(n_e\) | electron density | `electron_density` | cm\(^{-3}\), `(D,)` |
| \(n_{\rm pert}\) | neutral-collision proxy | `collision_density_proxy` | cm\(^{-3}\), `(D,)` |
| \(a_l\) | damping ratio | `damping_ratio` internally | dimensionless, `(D,L)` |
| \(u\) | absolute Doppler offset | `doppler_offset` | dimensionless |
| \(H(a,u)\) | dimensionless Harris/Voigt function | profile helper output | dimensionless |
| \(\kappa_{l,i}\equiv\kappa_{\nu,l}(c/\lambda_i)\) | line mass absorption sampled on the native wavelength grid; not a per-nm density | `line_mass_absorption_coefficient` | cm\(^2\) g\(^{-1}\), `(D,W)` |

`oscillator_strength` is exact source vocabulary, but its ordinary catalog
value is \(gf\), not bare \(f\). Chapter 6 must say this at first use. It must
not multiply a separate lower-level statistical weight into the production
formula a second time.

## 7. Integrated line strength

The normalized Doppler–Voigt representation is

\[
u=\frac{\nu-\nu_l}{\Delta\nu_{\rm D}},\qquad
\phi_l(\nu)=\frac{H(a_l,u)}{\sqrt{\pi}\Delta\nu_{\rm D}} .
\]

With

\[
\delta_l=\frac{\Delta\nu_{\rm D}}{\nu_l}
\]

the synthesis invariant

```python
CLASSICAL_LINE_STRENGTH_COEFFICIENT = 0.026538 / 1.77245
classical_line_strength = (
    CLASSICAL_LINE_STRENGTH_COEFFICIENT * oscillator_strength / frequency_hz
)
```

combines the parity-pinned approximation to
\((\pi e^2/m_ec)/\sqrt{\pi}\) with \(gf/\nu_l\). The per-depth pre-excitation
amplitude is then

```python
classical_line_strength * (
    partition_normalized_population / (mass_density * fractional_doppler_width)
)
```

and the lower-level excitation factor completes it.

Do not recompute the coefficient from newly rounded electron constants. The
literal `0.026538 / 1.77245` is part of the numerical method.

The atmosphere selected-line path is algebraically parallel but uses the
internal literal

```python
_CLASSICAL_LINE_STRENGTH_SCALE = (
    0.026538 / 1.77245 / LIGHT_SPEED_NM_PER_S
)
```

and multiplies it by wavelength, the packed strength lookup, and the stored
`partition_normalized_population_over_mass_density_and_fractional_doppler_width`.
Packed quantization is a Chapter 7 topic.

## 8. Lower-level population and FASTEX

The lower-level contribution is not stored as a public array. For one line,

\[
n_l f_{lu}
=
\frac{n_{s,r}}{U_{s,r}}
(gf)_l \exp[-E_l(hc/kT)] .
\]

The production kernels evaluate the exponential with a two-table approximation.

### 8.1 Atmosphere scalar and Numba rule

`build_fast_exponential_tables()` creates two NumPy `float64` arrays of length
1001:

```python
integer_step = exp(-arange(1001))
fractional_step = exp(-arange(1001) * 0.001)
```

`fast_exponential_lookup(x)` and
`_fast_exponential_lookup_compiled` use the same integer and rounded
millistep lookup. They return `0.0` for nonfinite \(x\), \(x<0\), or
\(x\ge1001\).

### 8.2 Synthesis Torch rule

`_fastex_tables(runtime_device, dtype)` constructs the same nominal samples,
then casts to:

- `float64` on CPU/CUDA;
- `float32` on MPS.

`fast_ex` uses the table only when \(x>0\) and `floor(x) < 1001`; outside that
table domain it evaluates `torch.exp(-x)` directly. It explicitly returns one
at \(x=0\).

The atmosphere and synthesis out-of-range rules are not identical. Physical
ordinary lines have nonnegative lower excitation in this chapter, but the
boundary tests must include \(x<0\), \(x=0\), a half-millistep, \(x\) just
below 1001, and \(x\ge1001\).

## 9. Doppler width

Chapter 3 already constructed and mapped
`fractional_doppler_widths`. Chapter 6 recalls one equation because the line
profile now gives it a physical consequence:

\[
\Delta v_{\rm D}^2
=\frac{2k_{\rm B}T}{m_l}+\xi^2,\qquad
\delta_l=\frac{\Delta v_{\rm D}}{c},\qquad
\Delta\lambda_{\rm D}=\lambda_l\delta_l .
\]

Exact atmosphere support:

```python
update_doppler_line_strength_factors(
    *,
    thermal_energy_erg,
    microturbulence,
    state,
)
```

- CPU NumPy `float64`;
- shape `(D,1006)`;
- major-isotope mass per packed slot;
- final packed slot remains zero;
- `microturbulence` is cm s\(^{-1}\);
- `thermal_energy_erg` uses the rounded atmosphere \(kT\) convention;
- the speed of light in the width division is
  `LIGHT_SPEED_CM_PER_S_EXACT`.

Exact synthesis support:

```python
compute_doppler_per_ion(temperature, microturbulence, atomic_masses)
```

- CPU NumPy `float64` before device upload;
- shape `(D,6,139)`;
- atomic columns 0–98, repeated across six stored ion stages;
- columns 99–138 remain zero in the atom-only construction;
- element masses rather than packed major-isotope masses;
- rounded `REFERENCE_BOLTZMANN_ERG_PER_K` and
  `REFERENCE_ATOMIC_MASS_GRAM`, exact synthesis light speed.

Chapter 6 does not recompute or remap those arrays. It selects the line's exact
slot and converts \(\delta_l\) to a wavelength/frequency width.

## 10. Natural, Stark, and van der Waals damping

The physical damping parameter is

\[
a_l=
\frac{\gamma_{\rm rad}+\gamma_{\rm S}n_e+
\gamma_{\rm vdW}n_{\rm pert}}
{4\pi\Delta\nu_{\rm D}} .
\]

The synthesis `LineCatalog` stores each damping field after division by
`12.5664 * line_frequency_hz`. The literal `12.5664` is the parity-pinned
decimal approximation to \(4\pi\); it must not be replaced by a newly evaluated
`4 * np.pi` when constructing the catalog fields. Therefore
`_accumulate_metal` evaluates the algebraically identical production form

```python
total_damping = (
    radiative_damping
    + stark_damping * electron_density
    + van_der_waals_damping * collision_density_proxy
)
damping_ratio = total_damping / fractional_doppler_width
```

The exact collision proxy, already constructed across Chapters 3–4, is

\[
n_{\rm pert}=
(n_{\rm H\,I}+0.42n_{\rm He\,I}+0.85n_{\rm H_2})
\left(\frac{T}{10^4\,{\rm K}}\right)^{0.3}.
\]

The three terms have distinct physical meanings:

- radiative damping is the finite lifetime of the upper/lower quantum states;
- Stark damping is collision/electric-field broadening scaled here by
  `electron_density`;
- van der Waals damping is neutral-perturber broadening scaled by the exact
  `collision_density_proxy`.

Damping redistributes opacity from core to wings. It must not change the
continuous-profile integrated strength in the ideal normalized limit.

Chapter 6 uses explicit nonzero damping values from the teaching row. The
missing-value default formulas in `_default_stark_log`,
`_default_van_der_waals_log`, and `_build_records` belong to Chapter 7.

## 11. Voigt and Harris profile

The Voigt profile is the convolution of:

- a Gaussian Doppler distribution from thermal and microturbulent velocities;
- a Lorentzian response from finite lifetime and collisions.

Payne Zero evaluates a Harris approximation rather than a general-purpose
complex error function.

### 11.1 Shared branch geometry

Both exact scalar/Torch full-profile evaluators use:

- a nearest 0.005-spaced table index for \(|u|\), clipped to 0…2000;
- low-damping branch for \(a<0.2\);
- asymptotic branch for \(a>1.4\) or \(a+|u|>3.2\);
- uncorrected asymptotic base for \(a>100\);
- `0.5642*a/u**2` tail for low damping and \(|u|>10\);
- an intermediate polynomial blend otherwise.

The inequalities are literal: \(a=0.2\) is not in the low-damping branch,
\(a=1.4\) is not sufficient by itself to enter the asymptotic branch,
\(a+|u|=3.2\) remains in the blend, \(|u|=10\) remains table-based, and
\(a=100\) still receives the asymptotic correction.

The branch comparisons, rounded coefficients, and table indexing are parity
state. No `fastmath` rewrite or generic Voigt replacement is allowed in the
exact implementation.

### 11.2 Atmosphere exact functions

```python
build_voigt_profile_basis() -> VoigtProfileBasis
evaluate_voigt_profile(
    frequency_offset: float,
    damping_parameter: float,
    basis: VoigtProfileBasis | None = None,
) -> float
```

Despite its argument name, `frequency_offset` is the dimensionless nonnegative
Doppler offset \(u\), not Hz. The compiled
`_voigt_profile_compiled` and `_voigt_profile_f64_compiled` transcribe the
same branch ordering for the selected and detailed paths.

### 11.3 Synthesis exact functions

```python
interpolate_harris_profile(
    doppler_offset,
    damping_ratio,
    harris_profile_h0_table,
    harris_profile_h1_table,
    harris_profile_h2_table,
)

harris_profile_at_line_center(...)
harris_wing_walk_profile(...)
```

`interpolate_harris_profile` is the full vectorized Torch evaluator.
The ordinary-metal production path has an additional low-damping shortcut:

- at line center and \(a<0.2\), it uses `1.0 - 1.128*a`;
- in the near wing and \(a<0.2\), it uses `H0 + a*H1`;
- beyond ten Doppler widths it uses `0.5642*a/u**2`;
- for larger damping it uses the full Harris evaluator.

The center shortcut, truncated walk, and float32 deposition mean that the
native discrete slab is not a second analytic normalized-profile definition.
The chapter must show the full Harris approximation first, then label the
ordinary-metal shortcut as an exact production optimization.

## 12. Atmosphere ordinary-line lane

### 12.1 Exact standard entry

The standard atmosphere line product is the selected-line call:

```python
accumulate_selected_line_opacity(
    *,
    selected_lines,
    opacity_wavelength_grid_nm,
    wavelength_bin_edges,
    continuum_line_selection_threshold,
    temperature,
    hc_over_kt,
    electron_density,
    ion_stage_populations_by_packed_slot,
    partition_normalized_population_over_mass_density_and_fractional_doppler_width,
    fractional_doppler_widths,
    wavelength_start_index=1,
    wavelength_stop_index=None,
) -> LineOpacityState
```

Exact reads:

The one-record `SelectedLineCatalog` has seven parallel arrays:

| exact field | fixture shape / dtype | Chapter 6 interpretation |
| --- | --- | --- |
| `packed_wavelength_index` | `(1,) int32` | logarithmically packed center coordinate |
| `packed_species_slot` | `(1,) int16` | packed species code; `abs(value)//10` is the one-based population/width slot |
| `lower_excitation_index` | `(1,) int16` | lookup index for lower excitation |
| `log_strength_index` | `(1,) int16` | lookup index for packed \(\log(gf)\) |
| `radiative_damping_index` | `(1,) int16` | lookup index for radiative damping |
| `stark_damping_index` | `(1,) int16` | lookup index for Stark damping |
| `van_der_waals_damping_index` | `(1,) int16` | lookup index for neutral damping |

These are opaque, provenance-bound fixture values for the same controlled Fe I
transition. Chapter 6 verifies their dtypes and decodes their physical effect
inside the pinned one-line kernel; it does not teach how the indices were
quantized, selected, serialized, or byte-swapped. The fixture-generation
command, conversion version, source-row identity, seven values, and fixture
hash must be recorded before use. `line_count` must be exactly one.

| input | axes / dtype at public call | unit / role |
| --- | --- | --- |
| `selected_lines` | one-record `SelectedLineCatalog` | already-selected packed line |
| `opacity_wavelength_grid_nm` | `(W,)` NumPy `float64` | input-only native atmosphere grid, normally `W=30000`; its construction is not reopened |
| `wavelength_bin_edges` | `(344,)`-policy boundary owned by Chapter 5 | packed continuum bins |
| `continuum_line_selection_threshold` | `(80,344)` NumPy, sanitized to `float32` | continuum-relative line cutoff |
| `temperature` | `(80,)`, sanitized NumPy `float64` | K |
| `hc_over_kt` | `(80,)`, sanitized to `float32` | cm |
| `electron_density` | `(80,)`, sanitized to `float32` | cm\(^{-3}\) |
| actual packed populations | `(80,1006)` NumPy `float64` | builds neutral collision proxy |
| line population/width support | `(80,1006)`, sanitized `float32` | \((n/U)/(\rho\delta)\), g\(^{-1}\) |
| packed fractional widths | `(80,1006)`, sanitized `float32` | dimensionless |

The function rejects any depth count other than 80 and requires actual packed
slot 840 (0-based) for H2 in the collision proxy.

Its public sanitization is part of the parity boundary:

- wavelength grid → contiguous `float64`; packed bin edges → contiguous
  `int64`;
- continuum threshold, electron density, and line
  population-over-density/width support → nonfinite values replaced by zero,
  clipped at the default `1e30` ceiling, contiguous `float32`;
- `hc_over_kt` and fractional Doppler widths follow the same policy with a
  `1e10` ceiling;
- actual packed populations become contiguous `float64` with nonfinite values
  replaced by zero;
- nonfinite or nonpositive temperature becomes `1.0` K before the neutral
  proxy is formed; the proxy then becomes sanitized `float32`.

Inside the compiled one-line route, an out-of-range continuum column, invalid
decoded species slot, a reconstructed center outside
`grid[start - 1] - 1 nm` … `grid[stop - 1] + 1 nm`, a TABLOG index outside
1…32768, a center walk beyond the grid, or a nonpositive Doppler width
suppresses that contribution. The controlled fixture must be valid on every
one of these dimensions; separate seam tests exercise the skip branches.

The Chapter 5 threshold is already

\[
q_\nu =
\frac{10^{-3}(\kappa_\nu^{\rm abs}+\sigma_\nu)}
{1-\exp(-h\nu/kT)}
\]

with the exact Chapter 5 floor, inactive-column policy, duplicated final
column, and packed sentinel. The division by stimulated emission is why the
atmosphere can compare a gross pre-stimulated line amplitude to this table and
apply the stimulated factor later in transfer. Chapter 6 consumes this
contract; it does not derive the 344-column grid again.

Exact writes:

- `LineOpacityState.line_mass_absorption_coefficient`: `(80,W)` NumPy
  `float32` for a nonempty production line route, cm\(^2\) g\(^{-1}\);
- `selected_line_count`: Python integer;
- no stimulated-emission factor yet.

`allocate_line_opacity_state` creates a `float64` zero slab, but a real
selected-line accumulation returns the compiled `float32` slab. The empty and
nonempty dtype difference must be tested, not silently normalized.

### 12.2 Exact one-line ordering

For the selected line, the compiled path:

1. maps the packed wavelength to a continuum-reference column;
2. maps the line center to the first atmosphere grid sample redward of the
   exact center;
3. decodes the packed species and log lookup indices;
4. constructs the parity-pinned classical strength;
5. tests the unexcited center strength against the continuum threshold;
6. applies FASTEX and tests again;
7. reads the fractional Doppler width;
8. forms radiative + electron Stark + neutral van der Waals damping and
   divides by the fractional width;
9. deposits the first grid point redward of the exact center and then at most
   100 additional red pixels, for at most 101 red-side deposits;
10. deposits the blue side beginning at the preceding grid point, for at most
    100 blue-side deposits;
11. includes the first point below the continuum cutoff, then stops;
12. respects those asymmetric 101-red/100-blue point caps.

This is the literal `range(center_index, center_index + 101)` versus
`range(1, 101)` behavior in pinned
`payne_zero_atmosphere/line_opacity.py:174-197`; “100 steps” must never be
shortened to “100 points each way.”

The exact serial call chain is
`_accumulate_selected_line_opacity_compiled` →
`_accumulate_selected_line_wings_compiled` →
`_voigt_profile_compiled`, with
`_fast_exponential_lookup_compiled` used before the wing call. The parallel
wrapper `_accumulate_selected_line_opacity_parallel` must remain dormant for
the one-record fixture.

Both center tests reject with `<`, so equality survives. The selected-line
wing walker uses its low-damping control path for `damping_parameter <= 0.2`;
at exactly 0.2 the table evaluator itself follows its non-low branch for
\(|u|\le10\), while the explicit far-wing expression is still used beyond
ten Doppler widths. Each deposited wing point is tested with `<` only after
deposit, so equality continues and the first strictly sub-threshold point is
retained.

Depths 8, 16, …, 80 are evaluated as gates before intervening seven-layer
blocks. This is exact standard selected-line behavior but its forest-scale
motivation belongs to Chapter 7. A one-line parity fixture must preserve it.

One line uses the serial cached `njit` kernel. `prange` activates only when
there is more than one line and more than one chunk; Chapter 6 must not invent
a one-line parallel speedup.

### 12.3 Detailed-transition route is secondary

`accumulate_transition_line_opacity` also sends types 0 and 3 through the
ordinary Harris path, in CPU NumPy/Numba `float64` work with `float32`
deposition and a 2000-step wing cap. It additionally carries blue selector,
hydrogen, autoionizing, and merged-continuum branches.

Chapter 6 may use a type-0, selector-zero one-record call as a diagnostic
cross-check, but it must label this as the optional detailed-transition route,
not the standard selected-line product. Its catalog decoding and all
nonordinary branches remain deferred.

### 12.4 Stimulated-emission boundary

`accumulate_transfer_range_compiled` later performs, per atmosphere frequency,

```python
line_mass_absorption_coefficient = (
    line_mass_absorption_coefficient_slab[:, frequency_index]
    * stimulated_all[frequency_index]
)
```

before the transfer moments use the line opacity. Chapter 6 may construct the
post-stimulated physical line for comparison, but the stored atmosphere
line-opacity product must remain labeled pre-stimulated.

## 13. Synthesis ordinary-line lane

### 13.1 Exact invariant construction

```python
precompute_invariants(
    catalog: dict,
    wavelength_grid_nm: np.ndarray,
    runtime_device=None,
) -> AtomicInvariants
```

For ordinary types 0 and 3 it:

- converts atomic number and ion stage to zero-based population indices;
- computes `frequency_hz = LIGHT_SPEED_NM_PER_S / wavelength_nm`;
- computes `classical_line_strength`;
- maps center and raw wing-anchor indices on the geometric wavelength grid;
- uploads line fields and Harris/FASTEX tables to the selected device.

The index functions are `nearest_grid_indices` and
`nearest_grid_indices_raw`. The first rounds on the logarithmic grid and emits
`-1`/`W` outside the two grid ends. The raw wing-anchor function reconstructs
the logarithmic origin and uses the exact
`origin_wavelength_nm - 64*np.spacing(origin_wavelength_nm)` guard before
`np.rint`; this prevents a platform-dependent one-pixel shift. It is parity
state, not a replaceable convenience.

Only a one-record catalog mapping is allowed in Chapter 6. Full catalog
parsing, window margins, sorting, and catalog caches stay in Chapter 7.

The mapping supplied to `precompute_invariants` contains the following
one-element per-line arrays:

| exact input key | shape / host dtype | unit / controlled value |
| --- | --- | --- |
| `line_type` | `(1,) int64` | `0`, ordinary |
| `atomic_number` | `(1,) int64` | `26`, Fe |
| `ion_stage` | `(1,) int64` | `1`, neutral stage |
| `wavelength_nm` | `(1,) float64` | nm, corrected/derived center |
| `index_wavelength_nm` | `(1,) float64` | nm, center/wing indexing coordinate |
| `oscillator_strength` | `(1,) float64` | dimensionless \(gf\) |
| `lower_excitation_cm` | `(1,) float64` | cm\(^{-1}\) |
| `radiative_damping` | `(1,) float64` | normalized by \(12.5664\nu_l\) |
| `stark_damping` | `(1,) float64` | cm\(^3\), normalized by \(12.5664\nu_l\) |
| `van_der_waals_damping` | `(1,) float64` | cm\(^3\), normalized by \(12.5664\nu_l\) |
| `raw_radiative_damping_log` | `(1,) float64` | retained source provenance |
| `raw_stark_damping_log` | `(1,) float64` | retained source provenance |
| `raw_van_der_waals_damping_log` | `(1,) float64` | retained source provenance |

“One-record catalog” describes the number of physical line records, not the
total number of arrays in the mapping. The same mapping must contain these
shared support entries:

| exact support key | shape / host dtype | controlled value / provenance |
| --- | --- | --- |
| `helium_line_type` | `(0,) int64` | explicitly empty because this catalog contains no helium line |
| `helium_line_center_cutoff_ratio` | scalar NumPy `float64` | exact `LINE_CENTER_CUTOFF_RATIO`, `1e-3` |
| `harris_profile_h0_table` | `(2001,) float64` | exact synthesis H0 table from the manifest-verified profile archive |
| `harris_profile_h1_table` | `(2001,) float64` | exact synthesis H1 table from the manifest-verified profile archive |
| `harris_profile_h2_table` | `(2001,) float64` | exact synthesis H2 table from the manifest-verified profile archive |

These five entries are required even with `do_helium=False` because
`precompute_invariants` unconditionally constructs the general
`AtomicInvariants` object. The pinned reads are
`payne_zero_synthesis/line_opacity.py:583-597`; the standard pipeline attaches
the Harris tables and empty helium metadata at
`payne_zero_synthesis/pipeline.py:176-182`, `:953-959`, and `:1622-1627`.
Omitting any support entry is not a reduced teaching API; it is an invalid
production call.

For the controlled zero-shift record, `index_wavelength_nm` equals
`wavelength_nm`. The three raw-log arrays are not ordinary-kernel inputs after
normalization, but retaining them prevents the one-record mapping from losing
source provenance and keeps the general invariant builder's empty
autoionizing branch reproducible.

The ordinary portion of the resulting `AtomicInvariants` is:

| exact invariant | shape for one line | dtype/device |
| --- | --- | --- |
| `wavelength_grid` | `(W,)` | work dtype, selected device |
| `n_wavelengths`, `grid_resolution` | scalars | Python integer / float |
| `metal_catalog_index` | `(1,)` | `torch.int64`, selected device |
| `metal_classical_strength` | `(1,)` | `torch.float32`, selected device |
| `metal_lower_excitation_cm` | `(1,)` | work dtype, selected device |
| `metal_radiative_damping` | `(1,)` | `torch.float32`, selected device |
| `metal_stark_damping` | `(1,)` | `torch.float32`, selected device |
| `metal_van_der_waals_damping` | `(1,)` | `torch.float32`, selected device |
| `metal_wavelength_nm` | `(1,)` | work dtype, selected device |
| `metal_population_ion_stage_index` | `(1,)` | `torch.int64`, selected device; value `0` |
| `metal_population_element_index` | `(1,)` | `torch.int64`, selected device; value `25` |
| `metal_center_index`, `metal_wing_index` | `(1,)` each | `torch.int64`, selected device |
| `metal_center_clamped`, `metal_wing_clamped` | `(1,)` each | `torch.int64`, selected device |
| `harris_profile_h0_table`, `harris_profile_h1_table`, `harris_profile_h2_table` | `(2001,)` each | work dtype, selected device |
| `exponential_integer_table`, `exponential_fraction_table` | `(1001,)` each | work dtype, selected device |

All `auto_*` tensors have length zero and the helium tensors are empty for the
controlled mapping. Their special-profile meanings are not introduced here.

### 13.2 Exact state reads

`accumulate_atomic(invariants, state, ...)` reads:

| exact state key | axes before selection | source dtype / upload |
| --- | --- | --- |
| `partition_normalized_populations` | `(D,6,139)` | host `float64` → work dtype/device |
| `fractional_doppler_widths` | `(D,6,139)` | host `float64` → work dtype/device |
| `mass_density` | `(D,)` | host `float64` → work dtype/device |
| `electron_density` | `(D,)` | converted to `DEFAULT_DTYPE` = `float32` |
| `temperature` | `(D,)` | work dtype/device |
| `hc_over_kt` | `(D,)` | work dtype/device |
| `collision_density_proxy` | `(D,)` | converted to `float32` |
| `continuum_opacity` | `(D,W)` | converted to `float32` |

The Chapter 5 background used here is

```python
continuum_opacity = continuum_absorption + continuum_scattering
```

computed by the progressive Chapter 5 implementation before opening any
Chapter 6 golden. No continuum process, edge triplet, or interpolation basis
is rederived.

Unlike the atmosphere reference table, this synthesis `continuum_opacity` is
the direct absorption-plus-scattering slab and is not divided by stimulated
emission. The exact synthesis kernel compares its pre-stimulated line
amplitudes with that slab and applies stimulation after deposition. This is a
pinned lane convention, not an invitation to make the two cutoff policies
look algebraically identical.

### 13.3 Dtype and device ownership

| quantity | CPU/CUDA | MPS |
| --- | --- | --- |
| wavelength grid, excitation, populations, widths, mass density | work `float64` | work `float32` |
| oscillator/classical strength and three normalized damping fields at invariant upload | `float32` | `float32` |
| electron density, collision proxy, continuum cutoff | `float32` | `float32` |
| line accumulation slab | `float32` | `float32` |
| stimulated-emission factor in `accumulate_atomic` | `float32` | `float32` |

`highp_dtype` chooses `float64` on CPU/CUDA and `float32` on MPS.
`ACCUMULATION_DTYPE` is always `torch.float32`.

### 13.4 Exact ordinary one-line ordering

`_accumulate_metal`:

1. selects the line's normalized population and fractional-width columns;
2. requires positive population, width, and mass density;
3. computes population divided by mass density and fractional width;
4. multiplies the `float32` invariant classical strength;
5. tests `pre_excitation_strength` against
   `LINE_CENTER_CUTOFF_RATIO * continuum_opacity`;
6. applies `fast_ex`;
7. tests the resulting `line_amplitude` against the same center cutoff;
8. forms total damping and divides by the fractional width;
9. forms the exact center shortcut/full-profile value;
10. deposits the center if it is on-grid;
11. normalizes the wing amplitude to the deposited center value;
12. determines per-depth reach from the continuum-relative cutoff;
13. deposits symmetric red/blue wings, with the exact low-damping
    near/far-wing branches;
14. casts deposited values to `float32`;
15. after the atomic component is complete, applies the wavelength-dependent
    stimulated-emission factor when `apply_stim=True`.

Both synthesis center tests use `>=`, so equality is active. The
low-damping center and near/far-wing shortcuts use `a < 0.2`; \(a=0.2\)
therefore follows the full Harris route.

The exact constants are:

```text
LINE_CENTER_CUTOFF_RATIO = 1e-3
WING_CUTOFF_FLOOR_RATIO = 1e-8
MAX_WING_PROFILE_STEPS = 1_000_000
NARROW_WING_MAX_REACH = 128
NARROW_WING_REACH_TIERS = (1, 8, 32, 128)
```

The exact cutoff arrays are

```python
center_cutoff = (
    continuum_opacity[:, metal_center_clamped]
    * LINE_CENTER_CUTOFF_RATIO
)
wing_cutoff_reference = maximum(
    continuum_opacity[:, metal_wing_clamped]
    * LINE_CENTER_CUTOFF_RATIO,
    continuum_opacity[:, metal_wing_clamped]
    * WING_CUTOFF_FLOOR_RATIO,
)
```

They are evaluated after the state continuum has entered as float32 and then
cast to the current work dtype inside `_accumulate_metal`.

The exact synthesis call chain is `accumulate_atomic` →
`_accumulate_metal`. Center deposits use `_scatter_add_rows`. With
`wing_mode="loop"`, every live line calls `_wing_walk_metal`. With the
default `wing_mode="batched"`, `_wing_reach_batched` determines per-depth
reach, `_wing_walk_narrow_batched` deposits lines of reach at most 128 through
the four tiers, and wider lines fall back to `_wing_walk_metal`.
`harris_wing_walk_profile` owns the low-damping near/far shortcut used by
those walks, and the batched deposit ultimately uses the same float32
accumulator through `_scatter_add_rows` / `_scatter_add_3d`. The implementation
must test both modes, but the main narrative should not turn this one-line
chapter into the Chapter 7 sparse-forest lesson.

The first below-cutoff step is retained by the reach convention. Batched
narrow and loop/wide paths must be parity-tested for the one-line fixture
before Chapter 7 composes many records.

### 13.5 Exact output

The public callable has the exact optional controls:

```python
accumulate_atomic(
    invariants,
    state,
    do_metal=True,
    do_helium=True,
    apply_stim=True,
    wing_mode="batched",
    output_line_mass_absorption_coefficient=None,
    host_accumulator=None,
) -> torch.Tensor
```

The standard call is:

```python
accumulate_atomic(
    invariants,
    state,
    do_metal=True,
    do_helium=False,
    apply_stim=True,
)
```

Chapter 6 leaves `wing_mode` at `"batched"` for the standard product and uses
`"loop"` only for an explicit parity comparison. It supplies neither output
buffer nor host accumulator. The source rejects `host_accumulator` when
helium or stimulated emission is enabled; no CPU accumulation detour belongs
in the taught device route.

For a one-record ordinary catalog:

- autoionizing arrays exist but have length zero;
- helium is not invoked;
- the returned tensor has `(D,W)` axes;
- unit is cm\(^2\) g\(^{-1}\);
- dtype is `torch.float32`;
- device is the invariant wavelength grid's device.

`do_metal=True` invokes both ordinary and autoionizing accumulators in the
general routine. The empty auto branch must be proved empty for this teaching
record rather than described as disabled by a nonexistent flag.

## 14. Scalar, NumPy, Numba, and Torch roles

| implementation | Chapter 6 purpose | exactness claim |
| --- | --- | --- |
| scalar Python equations | expose \(gf\), Boltzmann, width, damping, and normalized profile one factor at a time | analytic/dimensional reference |
| NumPy dense one-line helper | evaluate one readable profile over a local grid and all supplied depths | teaching implementation, no catalog/selection claims |
| atmosphere scalar Harris/FASTEX | inspect exact CPU tables and branches | exact scalar source translation |
| atmosphere serial `njit` one-record call | prove standard CPU line-product behavior | exact one-line atmosphere lane |
| Torch full Harris helper | compare the device table evaluator with its own scalar reference | exact synthesis profile lane |
| Torch `accumulate_atomic` one-record call | prove exact device state, cutoff, deposition, stimulation, and output dtype | exact one-line synthesis lane |

There is no pedagogical `prange` benchmark in Chapter 6. A single line does not
contain independent line chunks. The real atmosphere `prange` forest and
fixed-order chunk reduction belong to Chapter 7.

There is no artificial GPU claim for the atmosphere line path. Conversely,
the synthesis line slab stays on its Torch device; it is not converted to
NumPy inside the taught kernel.

## 15. Data and fixture plan

The Chapter 6 implementation now uses:

1. **Static inputs**
   - Chapter 3 mass tables;
   - atmosphere `line_opacity_tables.npz`, staged byte-identically with
     SHA-256
     `89f486122cb8939b23dc5423145a46d88a77df8daf57a1def35055b7b8205f16`;
   - synthesis `line_profile_tables.npz`, staged byte-identically with
     SHA-256
     `87b47fc76bed10455218f43c4b6686525b961002e72d6a5ef01255a08deb27d4`.
2. **One teaching subset**
   - `data/subsets/chapter06_fe_i_source_row_873702.npz`, SHA-256
     `bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04`;
   - the accepted raw ordinary record described in Section 5;
   - no full catalog duplication.
3. **Existing integration fixture**
   - `data/fixtures/chapter05_continuum_states.npz`, SHA-256
     `ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae`;
   - Chapter 6 may reuse its four six-depth schema-v4 synthesis states and
     recompute the Chapter 5 continuum background;
   - those states remain labeled upstream computed inputs, not constructed in
     Chapter 6.
4. **Required but not-yet-frozen atmosphere one-line integration fixture**
   - exactly 80 depths, because the public atmosphere line path rejects any
     other count;
   - packed actual/normalized population, width, density, temperature,
     `hc_over_kt`, continuum threshold, grid, and one already-selected record;
   - input-only role, with source/oracle command and hashes;
   - it must not be mixed into the teaching subset or a golden archive;
   - it remains explicitly unfrozen until the provenance-bound Fe I packing
     conversion, every quantized value and reconstructed wavelength, the
     80-depth input hash, and the serial-kernel oracle have been recorded and
     independently checked.
5. **Separate, not-yet-frozen goldens**
   - atmosphere pre-stimulated one-line slab and downstream stimulated view;
   - synthesis CPU-float64-work / float32-accumulation one-line slab;
   - backend-specific comparisons only after reader-built results exist;
   - neither golden may be called pinned until its producing command, input
     hashes, output hash, dtype, axes, and tolerance report are in the manifest.

The Chapter 5 golden is not an input. The continuum background is recomputed
from Chapter 5 code; goldens remain comparison-only.

## 16. Verification gates

### 16.1 Source and data identity

1. Verify the pinned commit and every source hash in Section 3.
2. Verify both table archives, keys, shapes, dtypes, byte sizes, and hashes;
   verify the generated 32768-entry TABLOG formula and its compiled `float32`
   cast.
3. Verify the two Harris bases remain distinct in the measured locations.
4. Verify the one-line subset is the declared raw source row and contains no
   extra catalog records.
5. Verify the candidate line has ordinary type 0, zero shifts/corrections, and
   three explicit damping values.
6. Verify the production center-gate visibility criterion means at least one
   active depth per regime, and recover the accepted counts `3/6`, `6/6`,
   `6/6`, and `6/6` in hot, solar, giant, and cool order.

### 16.2 Analytic and dimensional physics

7. Recover \(\lambda_l=10^7/|E_u-E_l|\) for the controlled row.
8. Recover \(gf=10^{\log gf}\) without a second statistical-weight factor.
9. Show that doubling \(gf\) doubles the integrated dense line opacity.
10. Show that doubling \(n_{s,r}/U_{s,r}\) doubles the line opacity.
11. Show that doubling \(\rho\) halves it.
12. Check the Boltzmann exponent is dimensionless.
13. Check the stimulated factor approaches \(h\nu/kT\) at low photon energy
    and one at high photon energy without replotting Chapter 5's derivation.
14. Check the continuous teaching profile normalization on a domain wide
    enough for each tested damping value; report measured tolerance.
15. Show that increased Doppler width lowers the core and broadens it while
    preserving the dense-profile area.
16. Show that increased damping redistributes opacity into the wings without
    changing the ideal integrated strength.

### 16.3 FASTEX and Harris branches

17. Check atmosphere scalar versus compiled FASTEX at zero, half-step ties,
    ordinary values, 1001 boundary, negative, and nonfinite inputs.
18. Check synthesis FASTEX against its own table/direct-exp branch contract on
    CPU `float64` and MPS-compatible `float32`.
19. Check atmosphere scalar versus both compiled Harris evaluators at
    \(a=0.2\), \(a=1.4\), \(a+u=3.2\), \(u=10\), and \(a=100\), including
    one-sided neighbors.
20. Check synthesis `interpolate_harris_profile` at the same branch seams
    against its own direct scalar transcription.
21. Check the synthesis low-damping center and near-wing shortcuts separately
    from the full Harris evaluator.
22. Do not assert atmosphere/synthesis bit identity. Record the measured
    cross-lane difference caused by their distinct H1 tables and shortcuts.

### 16.4 One-depth and all-depth physics

23. Reconstruct one selected depth's raw amplitude factor by factor.
24. Perturb only `electron_density` and show only the Stark part of the damping
    numerator responds.
25. Perturb only `collision_density_proxy` and show only the van der Waals part
    responds.
26. Perturb microturbulence and show both width and damping ratio respond in
    the correct directions.
27. Extend the same function across all depths and verify axes `(D,W)`, finite
    nonnegative values, and the declared outer-to-inner depth order.
28. Confirm different depths change the profile through population,
    temperature, density, electron density, collision proxy, and width rather
    than through hidden catalog mutation.

### 16.5 Atmosphere exact parity

29. Verify the one-record selected path requires 80 depths.
30. Verify one line chooses the serial cached `njit` path.
31. Verify red/blue center ownership, the maximum 101 red-side versus 100
    blue-side deposits, and first-below-cutoff inclusion.
32. Verify the exact 8-layer depth-gate result.
33. Verify the nonempty result is NumPy `float32` and the empty allocator is
    `float64`.
34. Verify the returned slab is pre-stimulated.
35. Multiply by the exact atmosphere stimulated array and compare with the
    transfer-kernel input view.
36. Treat the optional detailed-transition type-0 parity as a distinct route,
    never as the standard atmosphere product.

### 16.6 Synthesis exact parity

37. Verify the one record alone populates the ordinary metal invariant fields;
    auto and helium counts are zero.
38. Verify population-stage and element indices for Fe I are exact.
39. Verify both pre-excitation and post-FASTEX center cutoff decisions.
40. Verify all three damping terms and the final `damping_ratio`.
41. Verify center shortcut/full-Harris branch selection.
42. Verify loop and batched one-line wings under an explicit tolerance; where
    equality is exact, record it rather than assuming it.
43. Verify the first below-cutoff reach and `MAX_WING_PROFILE_STEPS` boundary
    with controlled cases.
44. Verify `apply_stim=False` gives the gross slab and `apply_stim=True`
    applies exactly one factor.
45. Verify result axes `(D,W)`, device, and `torch.float32` accumulation.
46. Compare CPU/CUDA work-float64 and MPS work-float32 under separately
    measured tolerances.
47. Measure the effect of casting one-line deposits to float32; do not hide it
    behind a float64 return conversion.

### 16.7 Handoff gates

48. Confirm the Chapter 5 continuum background is computed before opening a
    line golden.
49. Confirm Chapter 6 contains exactly one physical line record.
50. Confirm no H/He/autoionizing/molecular source table is consumed.
51. Confirm no catalog selection, many-line chunking, or race/reduction lesson
    appears before Chapter 7.

## 17. Exact traps and unresolved ambiguities

### Resolved traps

1. **`oscillator_strength` is \(gf\) for the ordinary source record.** The
   lower-level statistical weight is already included.
2. **`fractional_doppler_widths` is \(\Delta v_D/c\), not a wavelength in nm.**
3. **`frequency_offset` in `evaluate_voigt_profile` is dimensionless \(u\), not
   Hz.**
4. **The atmosphere and synthesis Harris tables are distinct.**
5. **The synthesis ordinary low-damping path is not simply
   `interpolate_harris_profile` at every point.**
6. **Atmosphere stimulation occurs downstream; synthesis stimulation occurs in
   `accumulate_atomic`.**
7. **The atmosphere direct line grid and synthesis logarithmic grid have
   different center/wing indexing.**
8. **Line accumulation is float32 in both production lanes even when surrounding
   work is float64.**
9. **The neutral perturber quantity is the Chapter 3/4 proxy, not total neutral
   density.**
10. **A single line does not justify `prange`; the real parallel unit is a
    line chunk.**

### Ambiguities to close before implementation acceptance

1. **Keep the teaching record unfrozen until every packing and oracle gate
   closes.** Raw row 873702 is source-clean and is still only the recommended
   Fe I candidate. A visibility result alone is insufficient. Freeze it only
   after the readable subset is manifest-bound, the six-depth synthesis
   visibility and golden are reproduced, the provenance-bound atmosphere
   packing conversion records every quantized value and reconstructed
   wavelength, and the 80-depth serial-kernel input and oracle hashes pass. If
   any gate fails, choose a new record by a documented rule and update this
   contract before staging data.
2. **The standard atmosphere and synthesis source catalogs are not proven to
   contain the same physical transition in the same representation.** The
   atmosphere fixture must therefore encode the same controlled Fe I teaching
   transition through a provenance-bound one-record conversion, not introduce
   a second physical line or imply an upstream catalog match. This proves the
   selected-line kernel against its pinned oracle; it does not justify
   atmosphere–synthesis slab equality. Chapter 7 may later establish a true
   source-catalog identity mapping.
3. **`LineTransitionCatalog.oscillator_strength` in the optional atmosphere
   detailed route is consumed as an already-scaled strength, despite its
   generic name.** Its exact unit/provenance must be settled in the Chapter 7
   catalog audit before reader prose discusses that route.
4. **Profile-normalization tolerances are not yet measured for the final
   teaching grid and damping cases.** The implementation pass must publish
   those measured values rather than say “machine precision.”
5. **Chapter 5 is now published and independently accepted.** Chapter 6 still
   depends on Chapter 5 code/output contracts and recomputes its continuum
   input before opening any line golden. The Chapter 5 golden remains
   comparison-only, not a Chapter 6 input.

## 18. Acceptance checklist

Chapter 6 is ready to implement only if all answers are yes.

- Is \(gf\) distinguished from \(f\) and from the lower-level population?
- Is the Chapter 3 normalized population consumed without reopening Saha or
  partition sums?
- Is the Chapter 5 stimulated factor reused exactly once?
- Are thermal and microturbulent widths connected to the already-stored
  fractional width without repeating Chapter 3's layout lesson?
- Are radiative, Stark, and van der Waals damping separately visible?
- Is the normalized continuous profile distinguished from the discrete
  deposited slab?
- Is sampled \(\kappa_{l,i}=\kappa_{\nu,l}(c/\lambda_i)\) distinguished from a
  per-nm density, with no Jacobian inserted into the opacity grid?
- Are the parity-pinned `0.026538 / 1.77245` and `12.5664` literals retained?
- Are the atmosphere and synthesis Harris authorities kept separate?
- Are FASTEX boundary differences explicit?
- Are axis, unit, dtype, device, and stimulation lifecycle stated before each
  production call?
- Does the atmosphere demonstration use one line and the serial compiled
  path?
- Have the Fe I record, packed 80-depth integration fixture, and both lane
  goldens passed their manifest, conversion, and oracle gates before any is
  called frozen?
- Does the synthesis demonstration retain float32 accumulation?
- Does the one-record synthesis mapping include the required empty helium
  metadata and all three manifest-verified Harris arrays?
- Are all catalog/selection/forest and special-profile topics deferred to
  Chapter 7?
- Are molecular records absent and deferred to Chapter 8?
- Does parity compare each reader-built lane with its own pinned oracle rather
  than assuming atmosphere–synthesis identity?
