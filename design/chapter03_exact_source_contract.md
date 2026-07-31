# Chapter 3 Exact Source and Data Contract

## Authority, identity, and scope

This contract is subordinate to `BIBLE.md` and
`design/global_chapter_contracts.md`. It audits the read-only development
oracles:

- Payne Zero source:
  `/Users/ysting/payne-zero`, commit
  `9c44001feae40b85146630499e6f8a5fed42e5af`;
- paper source:
  `/Users/ysting/Source_Files_Not_For_Review/main.tex`, SHA-256
  `e11507b9150550b246f6664debf22e540aa92d8261eb40daabb594da91bd8e0d`;
- paper PDF:
  `/Users/ysting/Source_Files_Not_For_Review/main.pdf`, SHA-256
  `5c3585794d31ac649eac13851d3a5a038d67a3f213b3218715f520e65a81b8ab`.

Neither oracle may become a reader runtime dependency. This document does not
modify either oracle and does not authorize a canonical chapter edit.

Chapter 3 answers one causal question:

> Given temperature, gas pressure, and composition at a depth, how are nuclei
> divided among excitation and ionization states, and what electron density
> makes that division consistent with charge conservation?

It owns:

- atomic levels, statistical weights, Boltzmann factors, and partition
  functions;
- adjacent-stage Saha equilibrium;
- Debye ionization-potential lowering and the associated occupation
  correction;
- the atom-only electron-density fixed point;
- total nuclei density, mass density, the neutral-collision proxy, fractional
  Doppler widths, and atomic specific internal energy;
- the atmosphere `(depth, 1006)` packed representation;
- the synthesis internal `(depth, 99, 6)` representation and public
  `(depth, 6, 139)` representation;
- actual ion-stage populations versus populations divided by partition
  functions;
- the full-electron-closure route versus the fixed-electron-density bridge;
- the exact serial, device-batched, and depth-parallel ordering boundaries.

It does **not** re-teach Chapter 2's array, Numba, Torch, checksum, or schema
foundations. It applies those foundations to the first substantial physical
state. It does **not** derive molecular mass action, residuals, Jacobians,
depth continuation, or molecular internal energy; those belong to Chapter 4.
It does not derive continuum opacity, line strength, damping, or a Voigt
profile; Chapters 5 and 6 consume the support fields produced here.

## Pinned source identities

| Source module | SHA-256 | Chapter 3 role |
| --- | --- | --- |
| `payne_zero_atmosphere/equation_of_state.py` | `719b316327fd6fb76dfa5267a2c9022d1384c269a0fa20d5b2ec733671ae3fa2` | Compiled partition/Saha stack, atomic closure, packed population fills |
| `payne_zero_atmosphere/population_layout.py` | `36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0` | Exact 1006-slot schedule |
| `payne_zero_atmosphere/runtime_state.py` | `fae240ec00f6f89d7c2a7ef721ce6e6539be234e523291fd6e8a096d731430e8` | Density, abundance, mass, and packed-state allocation |
| `payne_zero_atmosphere/doppler.py` | `e118a78bf5250ef5e1f77d652c9e78fbb7b92acf5c069f717faed7a3b3ea98f0` | Packed fractional Doppler widths and line-strength support |
| `payne_zero_atmosphere/specific_internal_energy.py` | `de06ba732ce1333d111a52223e39f5b4f80eece8cfc4ff2f30de9739e16d7ec5` | Atomic specific internal energy |
| `payne_zero_atmosphere/synthesis_bridge.py` | `142a960b5e710823754b02766803b3c1dd8c48c9945fdfabe560b4ee7e1acb50` | Packed-to-public cube mapping |
| `payne_zero_atmosphere/runner.py` | `05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7` | Exact atom-only, molecule-enabled, and fixed-\(n_e\) call order |
| `payne_zero_atmosphere/constants.py` | `ac1f1fbd345dc816eb3e70a8f97ebebc7a4c744fd2759b32ec19f8c88d987036` | Atmosphere exact/reference constant fence |
| `payne_zero_synthesis/equation_of_state.py` | `6497c29abb954e0b55d918cc22fa7b660952812c548faf1d7b1053345ef13562` | Torch partitions, Saha ladders, full and fixed-\(n_e\) states |
| `payne_zero_synthesis/ground_partition_table.py` | `6950686c89ea51e301b4b11256d9413dad58d82741a51580f3547aa012ade832` | Ordered ground-state partition corrections |
| `payne_zero_synthesis/pipeline.py` | `465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22` | Doppler cube, collision proxy, and fixed-\(n_e\) structured bridge |
| `payne_zero_synthesis/synthesis.py` | `590e430b6582fbcf601a52b721d8f65073432903773a99238073a1d821fe0d0c` | Engine bridge and mean nuclear mass |
| `payne_zero_synthesis/api.py` | `77718303c1e0052a520ece7fab277b3b1922c21d09b35a288596592d03310940` | Public `build_structured_atmosphere` boundary |
| `payne_zero_synthesis/constants.py` | `ed58004196790f9fb4a2871044c9cd36bf7bc42046a923f9314f7b8ea7456798` | Synthesis exact/reference constant fence |
| `payne_zero_synthesis/atmosphere_schema.json` | `2ba8d637e613be12ff43ce319a752616323f0341ea69f8e2391c3c244939777a` | Public cube names, shapes, and units |

The atmosphere and synthesis constant modules are deliberately independent.
Chapter 3 must not create a universal constant namespace. In particular, the
EOS computations use the rounded, parity-pinned values:

| Quantity | Atmosphere name | Synthesis name | Value |
| --- | --- | --- | --- |
| \(k_{\rm B}\), erg K\(^{-1}\) | `BOLTZMANN_ERG_PER_K_REFERENCE` | `REFERENCE_BOLTZMANN_ERG_PER_K` | `1.38054e-16` |
| \(k_{\rm B}\), eV K\(^{-1}\) | `BOLTZMANN_EV_PER_K_REFERENCE` | `REFERENCE_BOLTZMANN_EV_PER_K` | `8.6171e-5` |
| \(h\), erg s | `PLANCK_ERG_SECOND_REFERENCE` | `REFERENCE_PLANCK_ERG_SECOND` | `6.6256e-27` |
| atomic mass unit, g | `ATOMIC_MASS_GRAM_REFERENCE` | `REFERENCE_ATOMIC_MASS_GRAM` | `1.660e-24` |
| wavenumber per eV | `WAVENUMBER_PER_EV_REFERENCE` | `REFERENCE_WAVENUMBER_PER_EV` | `8065.479` |
| one-spin Saha coefficient | local `2.4148e15` term | `REFERENCE_SAHA_COEFFICIENT` | `2.4148e15` |

The light speed used with the rounded EOS constants is the exact
`2.99792458e10` cm s\(^{-1}\) value in both stacks. The raw line-catalog
conversion `8065.54429` is unrelated and must not enter this chapter's EOS.

## Exact source-interface index

The chapter does not need to show every signature at once, but its source
package and tests need to preserve this vocabulary.

### Atmosphere-side interfaces

| Exact name | Exact signature summary | Return/mutation |
| --- | --- | --- |
| `load_special_partition_tables` | `()` | cached `SpecialPartitionTables` |
| `load_iron_group_partition_grid` | `()` | `(7,56,10,9)` NumPy `float64` |
| `load_ionization_potential_table_cm` | `()` | `(999,)` NumPy `float64` |
| `load_packed_level_metadata` | `()` | `(6,365)` NumPy `int64` |
| `saha_partition_depth` | keyword-only physical scalars, `atomic_number`, `ion_stage_count`, `population_mode`, optional `charge_square_density_cm3` | NumPy `float64` vector |
| `saha_partition_depth_batch` | depth `temperature_k`, `electron_density_cm3`, fixed element/stages/mode, optional depth charge-square array | `(D,width)` NumPy `float64` |
| `iterate_electron_density` | keyword-only depth arrays, `AtmosphereRuntimeState`, `max_iterations=200`, `tolerance=1e-4` | mutates state, returns `None` |
| `populate_species` | keyword-only packed code/mode/output plus state and iteration controls | mutates output/state, returns `None` |
| `populate_all_species` | keyword-only depth/state/molecule/iteration controls | mutates both packed population arrays |
| `ion_stage_count_for_atomic_number` | `(atomic_number: int)` | stage count `int` |
| `atomic_population_slot_start` | `(atomic_number: int)` | zero-based start `int` |
| `decode_population_code` | `(code: float)` | `(atomic_number, output_count)` |
| `population_job_schedule` | keyword-only `include_molecules: bool` | ordered `list[PopulationJob]` |
| `build_runtime_state` | `(atmosphere: ModelAtmosphere)` | `AtmosphereRuntimeState` |
| `update_charge_square_density` | keyword-only `thermal_energy_erg`, `state` | mutated `(D,)` charge-square array |
| `compute_mean_nuclear_mass_amu` | `(elemental_abundances_by_layer)` | `(D,)` NumPy `float64` |
| `update_doppler_line_strength_factors` | keyword-only thermal energy, microturbulence, state | two `(D,1006)` arrays and state mutation |
| `compute_atomic_specific_internal_energy` | keyword-only `temperature_k`, `state` | `(D,)` NumPy `float64`, no state mutation |
| `structured_atmosphere_from_packed_state` | keyword-only physical columns, three packed arrays, abundances, optional molecular arrays/edge path | schema-shaped `dict[str,np.ndarray]` |
| `prepare_population_state` | `(config, *, temperature_iteration_index=1, setup=None, molecular_thermal_energy_erg=None)` | `AtmospherePopulationState` |
| `prepare_structured_handoff_population_state` | same signature as `prepare_population_state` | fixed-\(n_e\) `AtmospherePopulationState` |

`PopulationJob` fields are exactly `code`, `mode`, `start_slot`, and
`output_slots`; its `target` property selects one of the two exact packed
population field names.

### Synthesis-side interfaces

| Exact name | Exact signature summary | Return/mutation |
| --- | --- | --- |
| `EOSTables.from_npz` | `(path=_DEFAULT_INPUTS, device=None, dtype=None)` | device-resident `EOSTables` plus host decision arrays |
| `EOSTables.from_dict` | `(d: dict, device=None, dtype=None)` | same |
| `ground_partition_value` | `(label: int, temperature: float)` | scalar `float` |
| `ground_partition_values` | `(label: int, temperature: np.ndarray)` | NumPy `float64` array |
| `derived_state` | `(temperature, gas_pressure, electron_density, *, tables)` | device tensors plus host `float64` decision arrays |
| `populations` | `(state: dict, tables: EOSTables, n_elements=99, max_ion=6)` | `EOSResult` |
| `partition_functions_for_elements` | `(temperature, gas_pressure, electron_density, *, tables, elements, nion=1, apply_ground_partition=True)` | `dict[int,np.ndarray]`, arrays `(D,min(nion, available))` on CPU `float64` |
| `solve_electron_density` | physical columns, `*, tables`, optional mass/seed, `max_iter=200`, `tol=1e-4`, molecule controls | `ElectronDensityResult` |
| `solve_population_state` | same full-closure inputs | `PopulationState` |
| `solve_population_state_at_electron_density` | physical columns, `*, tables, electron_density`, optional mass and molecule controls | fixed-\(n_e\) `PopulationState` |
| `compute_doppler_per_ion` | `(temperature, microturbulence, atomic_masses)` | `(D,6,139)` NumPy `float64` |
| `compute_mean_nuclear_mass_amu` | `(elemental_abundances, atomic_masses=None)` | scalar `float` |
| `build_structured_atmosphere` | keyword-only physical columns, optional mean mass/microturbulence/mass/molecules/device/dtype/tolerance | public structured mapping |

`PopulationState` contains the exact named NumPy fields
`electron_density`, `total_nuclei_number_density`, `mass_density`,
`partition_normalized_populations`, `ion_stage_populations`,
`hydrogen_neutral_population`, `hydrogen_ionized_population`,
`hydrogen_partition_normalized_ion_stage_populations`,
`helium_neutral_population`, `helium_singly_ionized_population`,
`carbon_partition_normalized_ion_stage_populations`,
`magnesium_neutral_partition_normalized_population`,
`aluminum_neutral_partition_normalized_population`,
`silicon_neutral_partition_normalized_population`, and
`iron_neutral_partition_normalized_population`, plus `eos` and optional
molecular arrays. The atomic chapter leaves those optional molecular arrays
`None`.

## The physical spine

### A bound level is not an ion stage

For one ion stage \(r\) of element \(s\), let level \(i\) have excitation
energy \(E_i\) above that stage's ground level and statistical weight \(g_i\).
In LTE,

\[
\frac{n_i}{n_{s,r}}
=
\frac{g_i \exp[-E_i/(k_{\rm B}T)]}{U_{s,r}(T,\rho)},
\qquad
U_{s,r}
=
\sum_i g_i\exp[-E_i/(k_{\rm B}T)].
\]

The chapter must first use a two-level atom because it exposes all three
ideas without a table:

\[
\frac{n_1}{n_0}
=
\frac{g_1}{g_0}
\exp\!\left[-\frac{E_1-E_0}{k_{\rm B}T}\right].
\]

Every symbol must be defined in ordinary language. A useful first statistical
weight is \(g=2J+1\), described as the number of quantum states sharing the
same listed energy. The chapter need not teach a full angular-momentum course.

The reusable production quantity is not \(n_i\). It is

\[
\frac{n_{s,r}}{U_{s,r}},
\]

stored as `partition_normalized_populations`. A later line calculation
multiplies it by a level weight and Boltzmann factor. Chapter 3 may show that
one identity to establish meaning, but Chapter 6 owns the opacity formula.

### Adjacent ion stages

The paper's exact notation should be retained:

\[
\frac{n_{s,r+1}n_e}{n_{s,r}}
=
\frac{2U_{s,r+1}}{U_{s,r}}
\left(\frac{2\pi m_e k_{\rm B}T}{h^2}\right)^{3/2}
\exp\!\left[
-\frac{\mathcal{I}_{s,r}-\Delta\mathcal{I}_{s,r}}
{k_{\rm B}T}
\right].
\]

Here:

- \(n_{s,r}\) is the **actual** number density of ion stage \(r\), cm\(^{-3}\);
- \(n_e\) is electron number density, cm\(^{-3}\);
- \(U_{s,r}\) is dimensionless;
- \(\mathcal{I}_{s,r}\) is the isolated ionization energy, eV in the kernels;
- \(\Delta\mathcal{I}_{s,r}\) is its density-dependent lowering, eV.

The physical limiting predictions come before production code:

- low \(T\), large ionization energy, or large \(n_e\) favors the lower stage;
- high \(T\), small ionization energy, or small \(n_e\) favors the upper stage;
- all stages included in the internal normalization sum to one before
  abundance scaling.

The returned or stored stage subset can be smaller than that internal ladder.
Its sum is therefore at most one, not universally exactly one. The missing
fraction is an explicit truncation diagnostic, especially for hot iron-group
tests.

### Charge and particle closure

In the atom-only ideal-gas state,

\[
n_{\rm particle}
=
\frac{P_{\rm gas}}{k_{\rm B}T}
=
n_{\rm nuclei}+n_e,
\]

and charge conservation requires

\[
n_e
=
\sum_s\sum_r q_r n_{s,r}.
\]

This is a feedback loop because the Saha fractions on the right depend on
\(n_e\). The exact implementations use a damped fixed-point iteration, not a
Newton step: no derivative or Jacobian is formed.

## Partition-function production taxonomy

One smooth classroom formula does not replace the production partition
machinery. The chapter should teach the taxonomy in this causal order.

### 1. Explicit ordered level sums

The special light-element branches add level terms in a fixed loop order.
The stored energies are wavenumbers in cm\(^{-1}\), so the exponential uses

\[
\exp[-E_i hc/(k_{\rm B}T)].
\]

The shared special arrays cover H I, He I, He II, C I, C II, Mg I, Mg II,
Al I, Si I, Si II, Na I, O I, B I, and K I. The synthesis bundle additionally
contains Ca I and Ca II level arrays, although the normal
`_build_partition_state_for_element` route dispatches atomic numbers 20–28 to
the iron-group table before the Ca special cases can be reached.

The sums are ordinary ordered additions. The exact source does not use
log-sum-exp for these partition sums. A stable log-space derivation may be
used to explain large dynamic ranges, but it must not be labeled the exact
partition implementation.

### 2. Packed ordinary-ion interpolation

For ions without a special branch, packed integer metadata encodes two
partition values plus a decimal scale. Temperature is converted to a
stage-dependent bracket using the ionization energy. The packed arithmetic,
rounding test, and association order are part of parity.

Atmosphere source:

- `packed_level_metadata`: `(6, 365)`, `int64`;
- `ION_FRACTION_SCALE = [0.001, 0.01, 0.1, 1.0]`, `float64`.

Synthesis source:

- `packed_partition_table`: `(6, 374)`, `int64`;
- `partition_interpolation_scale`: `(4,)`, `float64`;
- the packed table remains as a host `int64` copy for discrete decisions.

These tables are **not byte-equivalent views of one table**. Across the first
365 columns, 18 integer entries differ, and synthesis has nine additional
columns. They must remain stage-specific assets.

Their availability dispatch also differs. The atmosphere stack gives special
six-stage blocks to C and N, while the synthesis stack reports six for C,
seven for N, and eight for O before the requested-count limit is applied.
Those extra stages can enter the synthesis normalization even when the public
cube stores only the first six. This is an algorithmic distinction, not
unused table padding.

### 3. Iron-group interpolation

Atomic numbers 20 through 28 use the PFIRON grid:

```text
(Debye-lowering node, temperature node, ion stage, element)
= (7, 56, 10, 9)
```

The values are dimensionless partition functions. The lowering coordinate is
in cm\(^{-1}\), with exact nodes:

```text
500, 1000, 2000, 4000, 8000, 16000, 32000
```

Temperature bracketing is piecewise in \(\log_{10}T\):

- spacing 0.02 below 3.7;
- spacing 0.03 from 3.7 through 4.0;
- spacing 0.05 above 4.0.

The temperature-bin index is bounded in both stacks, but the final
temperature interpolation weight is not generally clipped at extreme
temperatures. The exact code can therefore extrapolate from the edge
temperature pair.

The lowering edge policy must remain stack-specific:

- atmosphere uses the first plane below 500 cm\(^{-1}\), interpolates between
  interior planes, and clamps to the last plane at and above
  32000 cm\(^{-1}\);
- synthesis also uses the first plane below 500 cm\(^{-1}\), but above the
  last node it keeps the last two planes and an unclipped logarithmic weight,
  so it extrapolates.

The bounded lowering index in synthesis therefore does not imply a bounded
interpolation weight. This policy difference is parity-significant.

The public atmosphere helper is:

```python
def iron_group_partition_function(
    *,
    atomic_number: int,
    ion_stage: int,
    log10_temperature: float,
    lowering_energy_cm: float,
) -> float
```

It accepts only \(20\le Z\le28\) and ion stages 1 through 10.

### 4. Low-temperature ground correction

Only the synthesis EOS stack applies
`ground_partition_value`/`ground_partition_values` through
`_ground_partition_values_np`.

```python
def ground_partition_value(label: int, temperature: float) -> float

def ground_partition_values(
    label: int,
    temperature: np.ndarray,
) -> np.ndarray
```

The ground table is source-native ordered data:

```python
GROUND_PARTITION_TERMS[label] = (
    base_statistical_weight,
    ((weight_1, energy_cm_1), ...),
)
```

The exact module has 189 term records and a fallback label `666` with value
one. `FIRST_RANGE_LABELS` has 114 entries; `SECOND_RANGE_LABELS` has 168.
`_ground_partition_values_np` maps a six-stages-per-element helper slot to
those label ranges, with unity in unmapped gaps.

The correction is applied only where

\[
T < 2T_{\rm ref}, \qquad
T_{\rm ref} = \mathcal{I}\,\frac{2000}{11},
\]

and replaces the current partition by its maximum with the ground correction.
It is a floor, not an added excitation term.

Important implementation fact: `EOSTables` loads a tensor field named
`ground_partition_table` with shape `(605, 80)`, but the current runtime
ground correction does not index that tensor. It evaluates the ordered
`GROUND_PARTITION_TERMS` in `ground_partition_table.py`. The field must not be
presented as the active lookup merely because the dataclass contains it.

### 5. Debye lowering and occupation correction

The physical Debye relation is

\[
r_D
=
\sqrt{
\frac{k_{\rm B}T}
{4\pi e^2\,n_{q^2}}
},
\]

but the exact stacks preserve different rounded denominator literals:

```text
atmosphere:
sqrt(kT / (12.5664 * (4.801e-10)^2 * charge_square_density))

synthesis:
sqrt(kT / (2.8965e-18 * effective_charge_density))
```

The atmosphere `charge_square_density` is in cm\(^{-3}\) and includes the
electron contribution. The synthesis effective density is the pressure and
electron-density proxy described below. Replacing either source literal with
mathematical \(4\pi e^2\) changes the parity contract.

The per-unit-charge lowering is

\[
\Delta\mathcal{I}_1
=
\min\!\left(1,\frac{1.44\times10^{-7}}{r_D}\right)
\ \mathrm{eV}.
\]

The stage value is multiplied by the one-based `ion_charge` used by the
source ladder. The lowering is capped at 1 eV before that multiplication.

The initial proxy produced by `update_charge_square_density` is:

```text
q2 = 2 electron_density
excess = 2 electron_density - gas_pressure / thermal_energy_erg
if excess > 0:
    q2 += 2 excess
```

During the atom-only atmosphere closure it is replaced by

\[
n_e+\sum_{s,r}q_r^2n_{s,r}.
\]

The synthesis `_debye_lowering` uses the pressure/electron proxy on every
population evaluation; it does not carry the atmosphere kernel's explicit
stage-summed `charge_square_density`.

The occupation gates are branch- and stack-specific:

- an atmosphere special branch calls the correction whenever its returned
  `density_parameter > 0`; it does not apply the ordinary branch's
  \(T\ge4T_{\rm ref}\) and stage-lowering \(\ge0.1\) gates;
- an atmosphere ordinary branch first returns uncorrected below
  \(2T_{\rm ref}\), then requires nonzero statistical weight, stage lowering
  at least 0.1 eV, and \(T\ge4T_{\rm ref}\);
- synthesis applies its low-temperature block first. After that, a positive
  special occupation parameter bypasses the high-temperature and 0.1 eV
  gates; otherwise those gates and a positive statistical-weight override
  are required.

The correction may add an effective high-level count and the final partition
is floored at one. The chapter should explain why pressure dissolves high
levels, show one controlled gate crossing per policy that it claims, and then
point to the exact implementation. It should not reproduce hundreds of
special-case terms in Markdown.

## Atmosphere CPU/Numba contract

### Exact public and chapter-facing signatures

Module `payne_zero_atmosphere.equation_of_state`:

```python
def saha_partition_depth(
    *,
    temperature_k: float,
    electron_density_cm3: float,
    total_nuclei_number_density_cm3: float,
    elemental_abundance: float,
    atomic_number: int,
    ion_stage_count: int,
    population_mode: int,
    charge_square_density_cm3: float | None = None,
) -> np.ndarray
```

```python
def saha_partition_depth_batch(
    temperature_k: np.ndarray,
    electron_density_cm3: np.ndarray,
    atomic_number: int,
    ion_stage_count: int,
    population_mode: int,
    charge_square_density_cm3: np.ndarray | None = None,
) -> np.ndarray
```

```python
def iterate_electron_density(
    *,
    temperature_k: np.ndarray,
    thermal_energy_erg: np.ndarray,
    state: AtmosphereRuntimeState,
    max_iterations: int = 200,
    tolerance: float = 1.0e-4,
) -> None
```

```python
def populate_all_species(
    *,
    temperature_k: np.ndarray,
    thermal_energy_erg: np.ndarray,
    state: AtmosphereRuntimeState,
    molecules_enabled: bool,
    pressure_iteration_enabled: bool,
    temperature_iteration_index: int,
    temperature_iteration_cache: dict[str, int],
    molecular_state: MolecularEquilibriumState | None = None,
) -> None
```

`saha_partition_depth` is keyword-only. It returns a CPU NumPy `float64`
vector. The batch path returns contiguous `float64` with shape
`(depth, output_width)`. Each batch row calls the same compiled scalar kernel
and is intended to be bit-identical to the corresponding scalar result.

### Exact population modes

Let the base mode be `mode` for values up to 10 and `mode - 10` otherwise.
Modes 11–15 return all requested stages.

| Base mode | Meaning before caller abundance scaling |
| --- | --- |
| 1 | ion-stage fraction divided by partition function |
| 2 | actual ion-stage fraction |
| 3 | partition function |
| 4 | mean ionic charge contribution \(\sum_r r f_r\), in output slot 0 |
| 5 | 61-slot diagnostic block: partitions at the front and accumulated potentials beginning at slot 32 |

Thus:

- mode 11 supplies all-stage fractions divided by \(U\);
- mode 12 supplies all-stage actual fractions;
- mode 13 supplies all-stage partition functions.

`_fill_atomic_population_slice` converts modes 11 and 12 into number densities
by multiplying by

```text
total_nuclei_number_density
* elemental_abundances_by_layer[:, atomic_number - 1]
```

Mode 1, used for a selected stage, writes only output column 0 after the same
scale.

The apparently descriptive parameters
`total_nuclei_number_density_cm3` and `elemental_abundance` passed directly to
`saha_partition_depth` are intentionally discarded in that function. They
keep its call site explicit; population number-density scaling happens in the
caller. The chapter must not say the scalar Saha function itself returns
number densities.

### Direct-ratio Saha algorithm

The atmosphere kernel:

1. clamps `temperature_k` to at least 1 K and `electron_density_cm3` to at
   least `1e-40` cm\(^{-3}\);
2. builds partitions and ionization/lowering arrays;
3. forms adjacent-stage ratios directly in `float64`;
4. normalizes them with a reverse recurrence;
5. takes cumulative products to obtain stage fractions.

It does not use the synthesis stack's log-space normalization. Extreme
classroom values should therefore be used first in an analytic or stable
teaching helper, then production parity should be demonstrated only on the
declared physical range.

### Damped electron closure

At each depth the atmosphere path computes

```text
total_particle_density = gas_pressure / thermal_energy_erg
total_nuclei_number_density = total_particle_density - electron_density
```

It does not apply the synthesis path's `1e-300` positive floor to this
difference. The atmosphere runner is expected to supply a physically valid
state; a focused failure test should show that the EOS kernel is not itself a
complete seed validator.

It then loops over atomic numbers 1 through 99 in ascending order, obtains
mode-12 fractions, scales them to actual populations, and accumulates charge.
The charge sum uses the returned mode-12 stages. Internal helper stages that
participated in Saha normalization but were not returned do not contribute to
this fixed point. This retained-stage closure is the exact implemented model.
The update is exactly:

```text
raw = sum(ion_charge * ion_population)
bounded = max(raw, 0.5 * old_electron_density)
new = 0.5 * (bounded + old_electron_density)
relative_error = abs((old - new) / max(new, 1e-300))
```

This is a damped fixed point. Because of the lower bound followed by the
average, one iteration cannot reduce the old electron density below 75% of
its previous value.

After every update:

- `electron_density` is replaced by `new`;
- `total_nuclei_number_density` is recomputed;
- `charge_square_density` becomes the ionic charge-square sum plus \(n_e\);
- convergence is tested against `tolerance`;
- on convergence, `mass_density` is updated.

Failure after `max_iterations` raises `RuntimeError` with the depth index.

### Parallel boundary

For more than one depth, `iterate_electron_density` calls:

```python
@numba.njit(parallel=True, nogil=True, cache=True)
def _iterate_electron_density_parallel(...)
```

Its outer loop is `numba.prange(layer_count)`. Each iteration owns one depth
row, including that row's complete electron fixed point and ascending
99-element sum. There is no cross-depth reduction. This is a genuine safe
parallel boundary and the source claims bit-identical depth results relative
to the scalar layer implementation.

For one depth, the wrapper uses `_iterate_electron_density_layer`. The chapter
should compare:

- one-depth scalar reference;
- multiple one-depth scalar calls;
- the multi-depth `prange` route;
- one and several Numba thread counts.

The expected claim is exact equality if measured on the pinned environment,
because thread grouping does not alter an individual depth's arithmetic. It
must still be recorded as a measured claim rather than inferred from the
decorator.

`saha_partition_depth_batch` is compiled but serial in depth. It removes
Python boxing overhead; it is not a `prange` function.

### Full runner order

In `prepare_population_state`, the atom-only route is:

```text
build_runtime_state
→ update_charge_square_density
→ first population job triggers iterate_electron_density
→ cache marks that temperature iteration as solved
→ population_job_schedule fills actual and n/U packed arrays
→ update_doppler_line_strength_factors
```

The full schedule refreshes all population slices at the final electron
density. This matters because `iterate_electron_density` itself stores the
fractions evaluated immediately before its last damped update. Those
intermediate slot values can be slightly stale within tolerance; the
subsequent scheduled fill is the reusable final population state.

The molecule-enabled route is different:

```text
code=0 population request
→ Chapter 4 molecular equilibrium and electron closure
→ ordered population schedule
```

It does not use the atom-only `prange` closure. Chapter 3 should state this
boundary once and defer its mechanism to Chapter 4.

## Synthesis Torch contract

### Exact dataclasses and functions

Module `payne_zero_synthesis.equation_of_state`:

```python
@dataclass
class EOSTables:
    device: torch.device
    dtype: torch.dtype
    ...
```

```python
@dataclass
class EOSResult:
    partition_functions: torch.Tensor
    partition_normalized_populations: torch.Tensor
    ion_stage_fractions_over_partition: torch.Tensor
```

Each `EOSResult` field has shape `(depth, 99, 6)` and lives on
`tables.device` with `tables.dtype`.

```python
def populations(
    state: dict,
    tables: EOSTables,
    n_elements: int = 99,
    max_ion: int = 6,
) -> EOSResult
```

```python
def solve_electron_density(
    temperature,
    gas_pressure,
    elemental_abundances,
    *,
    tables: EOSTables,
    mean_nuclear_mass_amu=None,
    electron_density_seed=None,
    max_iter: int = 200,
    tol: float = 1e-4,
    molecules: bool = False,
    molecules_path=None,
) -> ElectronDensityResult
```

```python
def solve_population_state(
    temperature,
    gas_pressure,
    elemental_abundances,
    *,
    tables: EOSTables,
    mean_nuclear_mass_amu=None,
    electron_density_seed=None,
    max_iter: int = 200,
    tol: float = 1e-4,
    molecules: bool = False,
    molecules_path=None,
) -> PopulationState
```

```python
def solve_population_state_at_electron_density(
    temperature,
    gas_pressure,
    elemental_abundances,
    *,
    tables: EOSTables,
    electron_density,
    mean_nuclear_mass_amu=None,
    mass_density=None,
    molecules: bool = False,
    molecules_path=None,
) -> PopulationState
```

### Host/device precision islands

`EOSTables.from_npz`/`from_dict` uploads numeric tables to the selected device
and work dtype. It also retains host copies:

- packed partitions as `int64`;
- ionization potentials as host `float64`;
- packed interpolation scales as host `float64`;
- PFIRON values and both lowering grids as host `float64`.

Temperature brackets, lowering brackets, and occupation regime gates are
chosen on the host `float64` arrays. The complete PFIRON interpolation and
the complete ordinary packed-table interpolation—including weights and the
continuous interpolated values—also run in NumPy `float64`; their results are
cast to the selected Torch dtype and device. Special partition sums,
occupation additions, and the log-space Saha ladder use Torch tensors in the
selected work dtype and device. A chapter precision diagram must show these
host interpolation islands rather than describing all continuous partition
work as device arithmetic.

Default synthesis policy remains:

- CPU/CUDA: `torch.float64`;
- MPS: `torch.float32`.

`populations` loops over atomic number in ascending Python order. Work across
depth within a given element is vectorized on the selected device. There is no
`torch.compile` path and no Numba `prange` inside synthesis.

### Log-space Saha ladder

`_saha_ladder` forms adjacent ratios in log space:

```text
log_stage_ratio
→ cumsum over ion-stage axis
→ subtract per-depth maximum
→ exp
→ normalize over ion stages
→ divide by partition function
```

Its input and working layout is `(ion_stage, depth)`. `populations` transposes
to `(depth, element, ion_stage)` when storing the result.

The normalization includes up to two helper stages beyond the requested count,
bounded by table availability. The public result stores at most six stages.
Therefore “six ions for every element” is false: six is storage capacity, and
availability/requested-count rules decide which entries are meaningful or
zero. When the internal ladder has more stages than the stored subset, the
stored fractions need not sum to one.

### Synthesis full charge closure

`solve_electron_density`:

1. converts temperature and gas pressure to host NumPy `float64`;
2. sets \(P/(kT)\);
3. uses `0.5 * total_particle_density` when no seed is supplied;
4. replaces nonfinite or nonpositive seed entries by that default;
5. evaluates all-element Torch populations at the current \(n_e\);
6. returns stage fractions to host NumPy `float64`;
7. scales them by the depth-dependent abundance matrix and nuclei density;
8. accumulates charge from the six stored stages in fixed NumPy axis order;
9. applies the same lower bound and half-step damping as the atmosphere
   implementation;
10. stops only when **all** depths satisfy `tol`.

Any seventh or later helper stage used in `_saha_ladder` normalization is not
present in that six-stage charge sum.

Unlike the atmosphere `prange` route, already-small residuals are not frozen
while another depth continues. All depth rows are reevaluated until the
worst row converges. This changes final last digits and is another reason not
to claim atmosphere/synthesis cross-stack identity.

At exit, `ElectronDensityResult` contains:

| Field | Shape | Unit/dtype/device |
| --- | --- | --- |
| `electron_density` | `(D,)` | cm\(^{-3}\), NumPy `float64`, CPU |
| `total_nuclei_number_density` | `(D,)` | cm\(^{-3}\), NumPy `float64`, CPU |
| `mass_density` | `(D,)` | g cm\(^{-3}\), NumPy `float64`, CPU |
| `eos` | dataclass of `(D,99,6)` tensors | selected Torch dtype/device |

As in the atmosphere closure, `eos` is the population evaluation immediately
before the last damped update. The code does not perform an extra atom-only
EOS evaluation at the final updated \(n_e\). The residual and population
conservation checks must therefore use the declared tolerance; tests must not
silently “fix” the production result before comparing it to a golden.

### Actual abundances are applied in assembly

The full closure calls `populations` with a 99-element vector of ones. It then
applies the requested abundance matrix explicitly when accumulating charge.
`_assemble_population_state` applies the same abundance scale to the returned
fractions:

```text
total_nuclei_number_density[:, None]
* elemental_abundances
```

This avoids applying the abundance twice. A direct call to `populations` may
accept a state abundance vector, but the exact full-solver route uses unit
abundances internally.

## Full closure versus fixed-electron-density bridge

The fixed bridge is a separate physical claim, not a faster spelling of the
full solve.

`solve_population_state_at_electron_density`:

- preserves the supplied `electron_density`;
- computes
  `total_nuclei_number_density = max(P/(kT) - electron_density, 1e-300)`;
- always evaluates the atomic partitions and Saha fractions once at that
  density;
- accepts an exact supplied `mass_density`, or derives one if absent;
- in the atom-only route, directly assembles the public population bundle;
- in the molecule-enabled route, performs the additional ion-formation and
  molecular-equilibrium population work owned by Chapter 4, while retaining
  the supplied electron density;
- does not test or enforce charge conservation.

The public synthesis boundary is:

```python
def build_structured_atmosphere(
    *,
    temperature,
    column_mass,
    gas_pressure,
    electron_density,
    elemental_abundances,
    mean_nuclear_mass_amu: float | None = None,
    microturbulence=None,
    mass_density=None,
    molecular_lines: bool = True,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    eos_tolerance: float = 1.0e-5,
) -> dict[str, np.ndarray]
```

The exact engine call chain is:

```text
payne_zero_synthesis.api.build_structured_atmosphere
→ synthesis.build_structured_atmosphere_from_columns
→ pipeline.build_structured_atmosphere_from_columns
→ equation_of_state.solve_population_state_at_electron_density
```

`pipeline.build_structured_atmosphere_from_columns` has an
`electron_density_seed` argument, but in this call chain it is the supplied,
already-established electron density and is passed to the fixed-\(n_e\)
function. The word `seed` must not lead the chapter to describe a hidden
charge solve.

`eos_tolerance` is forwarded by the synthesis wrapper as the pipeline
argument `tol`, but the fixed-density call does not receive that argument.
If molecules are enabled, the current molecule-backed fixed-density helper
uses its own literal `tol=1e-4`. Thus the public `eos_tolerance` changes
neither the atom-only fixed fill nor that molecule solve in this audited call
chain.

The atmosphere-side exact fixed handoff is
`prepare_structured_handoff_population_state`. It sets
`pressure_iteration_enabled=False` during the packed fill, so the final
atmosphere electron density is not iterated again in the atom-only route.
When molecules are enabled, the function first invokes the Chapter 4
molecular-equilibrium closure, which updates runtime density arrays before
that fixed packed fill. The Chapter 3 preservation test must therefore use
`molecules_enabled=False`; it must not generalize the atom-only claim across
the deferred molecular branch.

Required check:

```text
output.electron_density == supplied electron_density
```

must be exact for the CPU NumPy field. A separate diagnostic may report the
charge residual, but the bridge must not alter the input to reduce it.

## Density, perturber, Doppler, and energy support

### Element and mass density

Atmosphere runtime construction uses:

```python
def compute_mean_nuclear_mass_amu(
    elemental_abundances_by_layer: np.ndarray,
) -> np.ndarray
```

and computes

\[
\rho
=
n_{\rm nuclei}\,\bar m_{\rm nuclei}\,m_u.
\]

The atmosphere function returns the direct weighted sum

```text
sum(elemental_abundance * reference_atomic_mass)
```

without dividing by an abundance sum. Its deck-derived H/He/heavy-element
representation is expected to be on the intended number-fraction scale.

The synthesis `_mass_density_from_composition`, by contrast, divides the
weighted mass by the abundance sum when it must infer the mean mass. The
public synthesis engine's `compute_mean_nuclear_mass_amu` does the same.
These definitions must not be merged. Supplying the converged atmosphere
`mass_density` to the fixed bridge preserves the atmosphere result and avoids
an unnecessary re-derivation.

### Neutral-collision density proxy

The exact ordinary-line collision proxy is

\[
n_{\rm pert}
=
\left(n_{\rm H\,I}
+0.42\,n_{\rm He\,I}
+0.85\,n_{\rm H_2}\right)
\left(\frac{T}{10^4\ {\rm K}}\right)^{0.3}.
\]

All population terms are actual number densities in cm\(^{-3}\), so the proxy
has cm\(^{-3}\). It is not the total neutral number density. Chapter 3 can
compute the atom-only limit with \(n_{\rm H_2}=0\). Chapter 4 later supplies
the molecular term; Chapter 6 explains how the proxy enters damping.

Atmosphere selected-line code derives the same expression from packed actual
slots:

- slot 0: H I;
- slot 2: He I;
- slot 840: H\(_2\).

The synthesis pipeline derives it from the named structured fields
`hydrogen_neutral_population`, `helium_neutral_population`, and
`molecular_hydrogen_population`.

Hydrogen-line profile code also creates separate temperature-scaled He and
H\(_2\) perturber columns. Those belong to the later specialized-profile
chapter and should not be expanded here.

### Fractional Doppler widths

The physical support identity is:

\[
\frac{\Delta v_{\rm D}}{c}
=
\frac{1}{c}
\sqrt{
\frac{2k_{\rm B}T}{m}
+\xi^2
}.
\]

Atmosphere exact function:

```python
def update_doppler_line_strength_factors(
    *,
    thermal_energy_erg: np.ndarray,
    microturbulence: np.ndarray,
    state: AtmosphereRuntimeState,
) -> tuple[np.ndarray, np.ndarray]
```

Both returned arrays are CPU NumPy `float64`, shape `(D, 1006)`:

- `fractional_doppler_widths`, dimensionless \(v/c\);
- `partition_normalized_population_over_mass_density_and_fractional_doppler_width`,
  with units g\(^{-1}\).

The atmosphere implementation uses `major_isotope_mass_amu` by packed slot.
It computes only the first `ion_slots - 1` columns; the final slot remains
zero. It writes both outputs back into `AtmosphereRuntimeState`.

Synthesis exact helper:

```python
def compute_doppler_per_ion(
    temperature,
    microturbulence,
    atomic_masses,
) -> np.ndarray
```

It returns `(D, 6, 139)` NumPy `float64`, fills atomic species columns 0–98,
uses one element mass for every ion stage of that element, and leaves the 40
non-atomic species columns zero. The atmosphere-to-synthesis bridge instead
maps the already-computed packed atmosphere widths. Those are distinct
supported paths, not guaranteed identical mass conventions.

Chapter 3 owns the support quantity and mapping. Chapter 6 owns frequency
width, damping ratio, and line-profile consequences.

### Atomic specific internal energy

Exact function:

```python
def compute_atomic_specific_internal_energy(
    *,
    temperature_k: np.ndarray,
    state: AtmosphereRuntimeState,
) -> np.ndarray
```

It returns `(D,)` CPU NumPy `float64` in erg g\(^{-1}\). It does not mutate
`state.specific_internal_energy`.

The atom-only energy per volume is:

\[
u_{\rm atom}
=
\frac{3}{2}(n_e+n_{\rm nuclei})k_{\rm B}T
+
\sum_j n_j
\left[
E_{{\rm ion},j}
+ k_{\rm B}T
\frac{\partial\ln U_j}{\partial\ln T}
\right].
\]

The exact source:

- forms cumulative ionization energies from the packed 999-entry potential
  table;
- evaluates mode-13 partitions at `T * 1.001` and `T * 0.999`;
- approximates the logarithmic partition derivative as

  ```text
  (U_plus - U_minus) / (U_plus + U_minus) * 1000
  ```

- sums packed slots 0 through 839 in increasing order;
- divides by `mass_density`.

The 840-slot limit is deliberate: this is atomic internal energy. Molecular
internal energy and the four convection perturbation states belong to
Chapters 4 and 12.

## Exact population representations

### Atmosphere runtime state

`AtmosphereRuntimeState` is a CPU NumPy `float64` state except for ordinary
Python/dataclass metadata:

| Field | Shape | Unit |
| --- | --- | --- |
| `gas_pressure` | `(D,)` | dyn cm\(^{-2}\) |
| `electron_density` | `(D,)` | cm\(^{-3}\) |
| `total_nuclei_number_density` | `(D,)` | cm\(^{-3}\) |
| `mass_density` | `(D,)` | g cm\(^{-3}\) |
| `charge_square_density` | `(D,)` | cm\(^{-3}\), charge-square weighted |
| `elemental_abundances_by_layer` | `(D,99)` | linear relative number abundance |
| `mean_nuclear_mass_amu` | `(D,)` | amu |
| `ion_stage_populations_by_packed_slot` | `(D,1006)` | cm\(^{-3}\), actual |
| `partition_normalized_populations_by_packed_slot` | `(D,1006)` | cm\(^{-3}\) per partition function |
| `specific_internal_energy` | `(D,)` | erg g\(^{-1}\) |
| `major_isotope_mass_amu` | `(1006,)` | amu |
| `fractional_doppler_widths` | `(D,1006)` when filled | dimensionless \(v/c\) |

`ION_STAGE_SLOTS = 1006`. This is a packed historical interface, not 1006
dense consecutive atomic stages.

### Exact packed atomic start

```python
def atomic_population_slot_start(atomic_number: int) -> int
```

returns a zero-based start:

```text
Z <= 30:
    1-based start = 1 + ((Z - 1) * (Z + 2)) // 2
Z >= 31:
    1-based start = 496 + (Z - 31) * 5
```

`decode_population_code(code)` converts the integer part to atomic number and
the fractional part to output count using:

```text
int(fractional_part * 100 + 1.5)
```

Therefore codes such as `1.01`, `2.02`, and `20.09` mean 2, 3, and 10 output
slots respectively. The exact decimal constants must be used; this is not an
API for arbitrary floating-point construction.

### Mode-12 versus mode-11 schedules

The two packed arrays intentionally use different schedules:

- mode 12: actual ion-stage totals used by charge and free-free physics;
- mode 11: stage totals divided by partition functions used by level/line
  physics.

Important differences:

| Element range | Mode-12 output count | Mode-11 output count |
| --- | ---: | ---: |
| H | 2 | 2 |
| He | 3 | 3 |
| Cl | 5 | 6 |
| Ar | 5 | 5 |
| K | 5 | 6 |
| Ca through Ni, \(Z=20\ldots28\) | 5 | 10 |
| Cu, Zn | 3 | 3 |
| \(Z=31\ldots99\) | 3 | 3 |

An audit of the atom-only schedule gives:

- 198 jobs total;
- 356 populated actual slots;
- 403 populated partition-normalized slots;
- atomic slot range 0 through 837;
- no overlap within either mode;
- gaps are intentional.

When molecules are enabled, selected later slots are filled, including H\(_2\)
at zero-based slot 840. Chapter 4 owns that schedule extension.

### Synthesis internal and public layouts

`EOSResult` uses `(depth, 99, 6)`:

```text
depth → atomic number 1..99 → ion-stage storage 0..5
```

`PopulationState` and schema v4 use `(depth, 6, 139)`:

```text
depth → ion-stage axis → species axis
```

In the atom-only bridge, the first 99 species columns receive atomic numbers
1–99 and the remaining 40 columns stay zero. That zero tail is not the
complete molecular address rule. Chapter 4 owns the exact release mapping:
selected molecular line-list species codes write normalized populations into
stage index 5 at code-derived columns across the public species axis. Atomic
`fractional_doppler_widths` use the same public shape.

The exact public names are:

- `ion_stage_populations`: actual cm\(^{-3}\);
- `partition_normalized_populations`: cm\(^{-3}\) per partition function;
- `fractional_doppler_widths`: dimensionless \(v/c\).

No chapter may call the second array a bound-level population, an ion
fraction, or an actual population.

### Packed-to-cube bridge

`payne_zero_atmosphere.synthesis_bridge._packed_atomic_cube` maps all three
packed arrays to `(D,6,139)` NumPy `float64` cubes. It uses the synthesis
module's mode-11 slot map for the exact common layout. It maps only atomic
numbers 1–99 and stages 1–6. Unmapped slots remain zero.

Named structured fields are copied from exact packed indices:

| Public field | Packed source |
| --- | --- |
| `hydrogen_neutral_population` | actual slot 0 |
| `hydrogen_ionized_population` | actual slot 1 |
| `helium_neutral_population` | actual slot 2 |
| `helium_singly_ionized_population` | actual slot 3 |
| `hydrogen_partition_normalized_ion_stage_populations` | normalized slots `0:2` |
| `carbon_partition_normalized_ion_stage_populations` | normalized slots `20:22` |
| `magnesium_neutral_partition_normalized_population` | normalized slot 77 |
| `aluminum_neutral_partition_normalized_population` | normalized slot 90 |
| `silicon_neutral_partition_normalized_population` | normalized slot 104 |
| `iron_neutral_partition_normalized_population` | normalized slot 350 |

`structured_atmosphere_from_packed_state` accepts any packed width of at least
351, although the canonical atmosphere runtime width is 1006. The chapter
should use 1006 for atmosphere parity and treat the minimum as an exporter
validation detail, not a new canonical layout.

## Exact data contract

### Atmosphere invariant tables

| Source file | SHA-256 | Bytes | Required arrays |
| --- | --- | ---: | --- |
| `atmosphere_tables/special_partition_tables.npz` | `7d737524aacda1cc2281e5b18ff49f240ca34665dbe6c96d4dd0f39db4aedd22` | 11,364 | 29-entry offsets plus 14 energy/weight pairs, all `float64` |
| `atmosphere_tables/iron_group_partition_tables.npz` | `137629dea64eca46f77ea3656c18305ade912a468d7eb27029544c0106cc3296` | 45,573 | `iron_group_partition_grid (7,56,10,9) float64` |
| `atmosphere_tables/ionization_potential_tables.npz` | `82a2e82f2015da02c3d2bce77ca5337aa2b9c4e23d8d6219da07895896ca8a50` | 8,292 | `ionization_potential_cm (999,) float64` |
| `atmosphere_tables/packed_level_metadata.npz` | `de5f17b6a9eaec1d1b07e96fd02ff014279cd8eaa9f976fefde0e2a153961bc3` | 17,816 | `packed_level_metadata (6,365) int64` |
| `atmosphere_tables/isotope_tables.npz` | `53c8d315fb53f1e051dc2752b028fc270d7c17a2c1042279c04ffcb750aef5c6` | 169,568 | `major_isotope_mass_amu (1006,) float64`; isotope records are not needed by this chapter's Doppler path |

Energies and ionization potentials are in cm\(^{-1}\); statistical weights
and partitions are dimensionless; masses are amu; offsets are one-based table
indices despite being stored as `float64` in the special bundle.

### Synthesis invariant tables

| Source file | SHA-256 | Bytes | Chapter 3 role |
| --- | --- | ---: | --- |
| `synthesis_tables/partition_saha_inputs.npz` | `0e235e7f1edecf39630690f4c68f4fc952f55785a08174562bc9575100fc4e27` | 719,700 | Synthesis EOS tables plus a bundled 80-depth state fixture |
| `synthesis_tables/atomic_masses.npz` | `d4739fef7e03964aea5a7b2604f9585fd9095c26c58f5b7d5d040aaafeb5d117` | 1,076 | `atomic_mass_amu (99,) float64` |

The synthesis EOS bundle contains:

- `packed_partition_table (6,374) int64`;
- `ionization_potential_cm (999,) float64`;
- `iron_group_partition_grid (7,56,10,9) float64`;
- two 7-entry lowering grids;
- `element_block_offsets (29,) int64`;
- `partition_interpolation_scale (4,) float64`;
- `ground_partition_table (605,80) float64`;
- all special level energy/weight arrays, including Ca I and Ca II;
- `temperature`, pressure, electron, derived thermal, abundance, and
  stage-count arrays for an 80-depth computed state fixture.

The book's four-role data policy forbids copying this mixed bundle wholesale
and then calling it one static table. The self-contained Chapter 3 build
should split it deterministically into:

1. a static synthesis EOS table archive containing the active invariant
   packed, ionization, PFIRON, scale, offset, and special-level arrays;
2. an integration fixture containing the bundled 80-depth thermodynamic state
   and its depth-specific `(605,80)` `ground_partition_table`.

Both products must record the source bundle hash and per-array hashes.
`EOSTables.from_dict` still requires a `ground_partition_table` key even
though the active runtime correction uses code-native terms. A local exact
constructor call may merge that manifest-bound fixture field into its input
dictionary, but the data ledger must state that the current source never
reads it during EOS evaluation. Do not move a depth-specific computed array
into `data/static` merely to make one constructor dictionary look uniform.

### Stage-specific tables may not be deduplicated

The atmosphere and synthesis PFIRON grids and shared special energy/weight
arrays are byte-equal array-by-array. That does **not** make the complete EOS
table stacks interchangeable:

- the packed partition tables differ;
- 615 of 999 ionization-potential entries differ exactly;
- the largest audited ionization-table discrepancy is at zero-based index
  428, where the atmosphere and synthesis source bundles differ by
  `-15471700.0` cm\(^{-1}\);
- synthesis has additional packed columns and special arrays;
- the runtime branch and ground-correction policies differ.

Each stage must therefore use its own manifest-bound inputs. A shared
classroom two-level table is allowed only as a teaching fixture, never as a
production replacement.

### Loader honesty

Atmosphere table loaders check file existence and required keys.
`load_iron_group_partition_grid` also checks the exact four-dimensional
shape; `load_packed_level_metadata` checks only that the first axis is six.
The synthesis `EOSTables.from_npz` constructor requires keys through ordinary
dictionary access but does not perform a complete checksum, unit, or shape
validation.

The textbook manifest and focused tests must supply those stronger claims.
Do not attribute them to the source constructors.

## Minimal self-contained fixtures

Every fixture must state that it is synthetic or extracted, list units and
axes, and carry a deterministic SHA-256.

### Fixture A: analytic two-level atom

No external table is needed:

```text
energy_cm = [0, 10000]
statistical_weight = [2, 4]
temperature = [3000, 6000, 12000] K
```

It supports:

- direct Boltzmann ratio;
- explicit partition sum;
- population conservation;
- low- and high-temperature predictions.

This is a controlled teaching model, not a source-table parity case.

### Fixture B: exact one-element source branches

Use manifest-bound source tables and a short depth vector:

```text
temperature = [3500, 5000, 10000, 30000] K
electron_density = 1e13 cm^-3 at each depth
positive gas pressures declared explicitly
charge_square_density = 2e13 cm^-3
```

Evaluate at least:

- H for special-level and occupation behavior;
- one ordinary packed-table element;
- Fe for PFIRON behavior;
- modes 11, 12, and 13.

Include one Fe lowering coordinate above 32000 cm\(^{-1}\). Its purpose is
to prove that the atmosphere path clamps to the last lowering plane while the
synthesis path extrapolates from the final two; it is not a cross-stack
equality case.

One audited warning justifies keeping both stage stacks: at 30,000 K under a
specific audited pressure/density setup, the atmosphere and synthesis H
partition functions differ materially. The chapter should show a measured
comparison only after recording all inputs; it should not teach one as the
other's approximation.

### Fixture C: atom-only closure

Use 4–8 depths spanning cool, solar, and hot conditions with:

- `temperature (D,)`, K;
- `gas_pressure (D,)`, dyn cm\(^{-2}\);
- positive `electron_density_seed (D,)`, cm\(^{-3}\);
- `elemental_abundances (D,99)` or `(99,)`, linear number abundance;
- atmosphere mean nuclear mass and synthesis mass policy recorded separately.

This fixture should be small enough for a scalar depth reference and large
enough to trigger atmosphere `prange`.

### Fixture D: layout sentinels

Do not use physical values. Fill each packed slot with a unique exact integer
sentinel, map it through `_packed_atomic_cube`, and verify:

- axes `(D,6,139)`;
- exact H, He, C, Mg, Al, Si, and Fe locations;
- zero unmapped cells;
- actual and partition-normalized arrays are never swapped;
- public species columns 99–138 remain zero in the Chapter 3 atom-only route;
- that zero-tail check is not described as the later molecular address rule.

The fixture must be labeled interface-only, not a physical atmosphere.

### Fixture E: support quantities

Use 2–4 positive depths with explicit:

- actual H I, He I, and zero H\(_2\) populations;
- mass density;
- microturbulence in cm s\(^{-1}\);
- major-isotope or element masses according to the tested path.

Check the collision proxy, \(v_D/c\), `population/(rho * width)`, and the
zero/sentinel columns.

## Recommended golden outputs

Keep atmosphere and synthesis goldens separate because their algorithms and
tables are separately certified.

### `chapter03_atmosphere_saha_outputs.npz`

Comparison-only arrays:

- mode-11, mode-12, and mode-13 outputs for H, an ordinary element, and Fe;
- explicit `charge_square_density_cm3`;
- scalar and batch results;
- table hashes and source commit.

Expected exact claim: scalar row equals batch row bit-for-bit in the pinned
CPU/Numba environment.

### `chapter03_atmosphere_atomic_state.npz`

Comparison-only arrays:

- final `electron_density`;
- `total_nuclei_number_density`;
- `charge_square_density`;
- `mass_density`;
- full actual and partition-normalized `(D,1006)` arrays after
  `populate_all_species`;
- packed fractional Doppler widths;
- atomic specific internal energy.

Record Numba version, thread count, cold/warm state, and whether the full
population refresh occurred after closure.

Expected exact claim: scalar-depth and `prange` results are bit-identical only
after it is measured under fixed inputs and source tables.

### `chapter03_synthesis_atomic_state_cpu_float64.npz`

Comparison-only arrays:

- full-closure `electron_density`, nuclei density, and mass density;
- fixed-\(n_e\) state from the same supplied density;
- `EOSResult` partitions and fractions;
- public actual and partition-normalized `(D,6,139)` cubes;
- public Doppler cube;
- named neutral/ion fields.

Record the exact Torch version and device. CPU/CUDA float64 and MPS float32
need separate tolerance profiles; do not use one universal golden tolerance.

### `chapter03_packed_bridge_outputs.npz`

Comparison-only outputs from the exact packed-to-cube bridge, including the
named scalar population fields and unchanged fixed electron-density column.

The golden is loaded only after the reader-built mapping and physical checks.

## Verification ladder for Chapter 3

1. Two-level Boltzmann ratio and partition sum.
2. Internal complete-ladder fractions sum to one; a returned/stored subset is
   nonnegative, sums to at most one, and reports its truncation deficit.
3. \(n_{s,r}=U_{s,r}(n_{s,r}/U_{s,r})\) wherever \(U>0\).
4. Cool-neutral and hot-ionized Saha predictions.
5. Partition functions remain at least one in production paths.
6. Source table keys, shapes, dtypes, hashes, and units.
7. Atmosphere scalar versus batch Saha identity.
8. Atmosphere one-depth scalar closure versus depth-parallel closure.
9. Charge residual under the declared solver tolerance.
10. Element population sum versus abundance-scaled nuclei density, with the
    helper-stage truncation deficit reported rather than silently treated as
    a conservation failure.
11. Mass-density dimensional and arithmetic identity.
12. Collision-proxy direct arithmetic.
13. Doppler thermal and microturbulent limiting behavior.
14. Packed sentinel map and public cube axes.
15. Full closure versus fixed-\(n_e\): one closes charge; one preserves input.
16. CPU float64 synthesis component parity.
17. CUDA float64 and MPS float32 comparisons under measured backend-specific
    tolerances.

No check should report “machine precision” without naming the arrays, dtype,
device, and measured tolerance.

## Source and implementation traps

These are acceptance-critical.

1. **The two EOS stacks are not interchangeable.** They differ in packed
   tables, ionization potentials, available helper stages, ground corrections,
   Debye state, special-branch temperature gates, normalization, and closure
   stopping behavior.
2. **`saha_partition_depth` ignores two descriptive inputs.**
   `total_nuclei_number_density_cm3` and `elemental_abundance` are not used by
   the kernel; callers apply the abundance scale.
3. **The electron solver is not Newton.** Both atom-only implementations use a
   damped fixed-point update without derivatives.
4. **Final closure populations are one evaluation behind the final update.**
   The atmosphere full schedule refreshes them; the synthesis atom-only full
   solver does not perform an extra final EOS call.
5. **The atmosphere and synthesis depth stopping policies differ.** Atmosphere
   depths break independently inside `prange`; synthesis continues all depths
   until the worst row passes.
6. **`ground_partition_table` is loaded but not the active runtime lookup.**
   The current correction evaluates ordered code-native terms.
7. **Ground corrections are synthesis-only in the audited stacks.** Do not
   imply the atmosphere packed kernel applies the same floor.
8. **Special H/He temperature behavior differs.** Synthesis gates explicit H I
   excited levels below 9000 K, He I below 15000 K, and He II below 30000 K;
   the atmosphere special kernel sums its listed terms without those same
   gates.
9. **The partition sums are direct ordered additions.** The synthesis Saha
   ladder is log-stable; that does not make the special partition sums
   log-sum-exp calculations.
10. **PFIRON edge behavior is stack-specific.** Both stacks can extrapolate
    in temperature. Above the last lowering node, atmosphere clamps to the
    last plane while synthesis extrapolates from the last two planes because
    its lowering weight is not clipped.
11. **The 1006-slot arrays are sparse interfaces.** They are not 1006 atomic
    ion stages, and actual and normalized schedules differ.
12. **The public cube axes are not the internal EOS axes.**
    `(D,99,6)` and `(D,6,139)` must be named explicitly every time they meet.
13. **Six is storage capacity, not a universal stage count.** Table
    availability and requested counts control populated stages.
14. **`partition_normalized_populations` is neither a fraction nor a bound
    level population.** Its public unit is cm\(^{-3}\) per partition function.
15. **The fixed-\(n_e\) bridge does not close charge.** It preserves the
    supplied atmosphere electron density by design.
16. **Mean nuclear mass conventions differ.** Atmosphere takes a direct
    weighted sum; synthesis normalizes by the abundance sum when deriving it.
17. **Doppler mass conventions differ by path.** Packed atmosphere uses
    major-isotope mass per slot; synthesis column building uses element masses.
18. **The neutral-collision quantity is a proxy.** It has exact H/He/H\(_2\)
    coefficients and temperature scaling; it is not a raw neutral density.
19. **Atomic specific internal energy is not filled by every population
    preparation call.** Its helper returns an array and covers only slots
    0–839.
20. **The synthesis EOS source bundle mixes data roles.** Split tables from
    its 80-depth fixture in the textbook repository.
21. **Source constructors are not full manifest validators.** Key access or a
    partial shape check does not prove provenance or physical validity.
22. **Mode codes are encoded decimals.** Preserve exact code constants and the
    source decoder; do not invent string parsing or a renamed enumeration.
23. **Molecule-enabled ordering is deferred.** It does not use the atom-only
    `prange` closure, can update the atmosphere handoff density state before a
    fixed packed fill, and Chapter 3 must not pre-teach Chapter 4's solver.
24. **Public builder `eos_tolerance` does not authorize a full charge solve.**
    In the exact column bridge, populations are evaluated at fixed supplied
    \(n_e\). The wrapper forwards this value to the pipeline as `tol`, but the
    fixed-density call does not receive it. Atom-only work has no electron
    iteration, and the current molecule-backed helper hardcodes `tol=1e-4`.
25. **The nuclei-density floors differ.** Synthesis clips
    \(P/(kT)-n_e\) to `1e-300`; the atmosphere atom-only closure uses the raw
    difference and relies on valid upstream state.
26. **Stored stage sums are not universally one.** Both stacks may normalize
    over helper stages that are not returned or stored. Equality to the
    elemental abundance scale is a controlled-regime check; the general
    diagnostic is a nonnegative truncation deficit.

## Reader-facing source treatment

The canonical progressive package should contain exact, manifest-verified
definitions. The chapter should never paste either 1900–2300-line EOS module
into Markdown.

Good visible code cards are:

1. a 10–15-line two-level Boltzmann function;
2. one exact ordered special-level sum excerpt;
3. the exact short `ground_partition_value`;
4. a readable adjacent-stage log-ratio helper labeled pedagogical;
5. the exact damped electron update extracted in context;
6. the exact `prange` outer boundary plus a source-parity result for the
   private body;
7. `compute_mean_nuclear_mass_amu`;
8. the short Doppler formula;
9. `atomic_population_slot_start` and `decode_population_code`;
10. a short exact packed-to-cube mapping excerpt;
11. the exact fixed-\(n_e\) call boundary.

Long special dispatches, packed interpolation, full table constructors, and
the 198-job schedule live in source. The narrative introduces their physical
need, displays a compact representative branch, executes the canonical
function, and interprets its actual output.

## Proposed 15-movement causal flow

This is a 12–15-section chapter structure; it contains no detached exercises.
Useful predictions and failure cases are resolved in the main sequence.

### 3.1 The missing electrons

Open with two layers having the same elemental mixture but different
temperature. Ask why pressure and composition alone do not determine H I,
H II, and \(n_e\). Show one single-panel ion-fraction preview without yet
claiming how it was computed.

**Produces:** the need for excitation, ionization, and charge closure.

### 3.2 The chapter state contract

State LTE, ideal-gas atom-only scope, outer-to-inner depth order, CGS units,
CPU atmosphere state, Torch synthesis state, and exact input/output shapes.
Use Chapter 2 contracts without reteaching broadcasting or devices.

**Produces:** the declared reads and writes.

### 3.3 Why one energy has several states

Introduce a level, excitation energy, and statistical weight with a two-level
ladder schematic. Derive and compute the Boltzmann ratio. Predict low- and
high-\(T\) behavior and inspect the actual values immediately.

**Produces:** level occupation ratios.

### 3.4 The partition function is a weighted inventory

Normalize the two-level populations, define \(U\), and show why the reusable
quantity is \(n/U\). Use the exact names
`partition_normalized_populations` and `ion_stage_populations` only after the
reader has constructed their distinction.

**Produces:** one checked excitation distribution and the partition identity.

### 3.5 Real atoms need more than one partition recipe

Move from the toy sum to:

```text
special ordered levels
or packed ordinary interpolation
or PFIRON interpolation
→ optional occupation/ground correction
→ U
```

Use one original schematic and three small numerical cases, not a source
dump. Explain exact/reference constants at the first production calculation.

**Produces:** checked special, ordinary, and iron-group partitions.

### 3.6 Removing an electron: the Saha ratio

Build the adjacent-stage equation from thermal phase space, ionization cost,
partition ratio, and electron crowding. Define every symbol. Compute H at
cool, solar-like, and hot conditions. Inspect stage conservation.

**Produces:** one-element multi-stage fractions.

### 3.7 Dense plasma lowers the last bound levels

Introduce Debye screening, ionization-potential lowering, and occupation
correction as one causal story. Show the 1 eV cap and one gate crossing. Place
the atmosphere/synthesis policy differences in a concise exact-boundary box.

**Produces:** density-corrected partitions and Saha ratios.

### 3.8 One element exposes the electron feedback

Show that the Saha fractions need \(n_e\) while charge conservation determines
\(n_e\). Iterate one element by hand for several steps, then introduce the
exact damping and convergence residual. Call it a fixed point.

**Produces:** a converged one-depth electron density.

### 3.9 Ninety-nine elements close one depth

Scale actual stage fractions by nuclei density and abundance, sum charge in
atomic-number order, and check charge closure. Compare the retained element
sum with its abundance scale and report any helper-stage truncation deficit.
Explain the last-evaluation-versus-final-update nuance before reporting
tolerances.

**Produces:** one complete atom-only depth state.

### 3.10 Depths are independent here

Show the exact atmosphere `prange` boundary: each depth owns a complete
fixed-point orbit. Compare scalar, batch, one-thread, and multi-thread outputs.
Contrast this with the serial batch Saha helper and the synthesis
device-batched element loop. Do not reteach Numba syntax from Chapter 2.

**Produces:** a checked multi-depth atom-only state.

### 3.11 From particle counts to density and broadening support

Compute nuclei density and mass density. Then derive the exact neutral
collision proxy in the atomic limit and the fractional Doppler width. State
that H\(_2\) is still zero because Chapter 4 has not built molecules.

**Produces:** `mass_density`, the atomic-limit collision proxy, and Doppler
support.

### 3.12 Where ionization energy is stored

Develop atomic specific internal energy as translation plus ionization plus
partition excitation. Show the symmetric 0.1% partition derivative and the
840-slot boundary. Defer convection's use of this state to Chapter 12.

**Produces:** `specific_internal_energy` in erg g\(^{-1}\).

### 3.13 Two exact layouts for two engines

Use an original side-by-side schematic:

```text
atmosphere: (D,1006) sparse packed slots
                  ↓ exact slot map
synthesis:  (D,6,139) public stage/species cube
```

Run sentinel mapping first, then map physical populations. Interpret exact H,
He, C, Mg, Al, Si, and Fe fields. Keep the unused atom-only tail 99–138
visibly empty, without presenting it as the later molecular address rule.

**Produces:** checked packed and public population states.

### 3.14 Full closure or fixed-\(n_e\): choose the claim

Run the full synthesis electron solve and the fixed-\(n_e\) bridge from the
same controlled columns with molecule work explicitly disabled for this
Chapter 3 comparison. The full path should reduce the retained-stage charge
residual; the fixed path should preserve the supplied electron density
exactly. Present a small claim table rather than calling either universally
better.

**Produces:** both exact route contracts and a parity/tolerance record.

### 3.15 Chapter summary and causal handoff

Answer the opening question using only computed results. List:

- actual atomic ion-stage populations;
- populations divided by partition functions;
- electron, nuclei, and mass densities;
- Doppler and atomic-energy support;
- exact packed and public layouts;
- which route did or did not close charge.

Then state the unresolved physical problem:

> Atomic conservation assumes every nucleus remains in an atom or ion. In
> cool layers, molecules bind several elements at once, so element-by-element
> Saha closure is no longer sufficient.

End with one direct link to Chapter 4. Introduce no new term in the summary.

## Original visual and plot plan

### Schematic 1: level ladder to charge closure

One landscape composition:

```text
energy levels + weights
→ partition sum
→ ion-stage ladder
↔ electron-density charge closure
```

Use short labels, distinguish actual \(n\) from \(n/U\), and show the feedback
arrow only around Saha/charge closure. It must be a new textbook-owned
composition in the established white/slate/navy/beige aesthetic.

### Schematic 2: two population layouts

Show a sparse 1006-slot atmosphere strip mapping into a six-by-139 synthesis
slice. Use highlighted H, He, Fe, an intentional gap, and one empty molecular
reservation. Do not imply every packed slot maps.

### Quantitative plots

Prefer successive one-panel figures:

1. excited-to-ground ratio versus temperature for the two-level atom;
2. H I/H II fractions versus temperature at one declared \(n_e\);
3. electron fixed-point residual versus iteration for one depth;
4. optional fractional Doppler width versus temperature for two masses.

Every figure must be generated from canonical chapter code, use the shared
professional style, label units, and be interpreted with actual numeric
values in the next paragraph. A layout schematic is not parity evidence.

## Local progressive-package requirements

The Chapter 2 progressive package does not yet contain the Chapter 3 EOS
modules. Before Chapter 3 claims self-contained execution, it needs
manifest-bound local implementations for:

- atmosphere `equation_of_state.py`;
- `population_layout.py`;
- `runtime_state.py`;
- `doppler.py`;
- `specific_internal_energy.py`;
- the necessary packed-to-cube definitions from `synthesis_bridge.py`;
- synthesis `equation_of_state.py`;
- `ground_partition_table.py`;
- the exact `compute_doppler_per_ion` and fixed-\(n_e\) bridge definitions or
  their complete dependency-bearing modules.

The preferred source strategy is:

- copy complete exact dependency-bearing modules when a long private call
  tree makes selective extraction fragile;
- use exact top-level definition extraction only when the dependency boundary
  is small and the source verifier checks AST identity;
- extend `src/PAYNE_ZERO_SOURCE_MANIFEST.json`;
- extend `scripts/verify_pinned_source_fragments.py`;
- keep table files under role-separated `data/static`, fixtures under
  `data/fixtures`, and comparison outputs under `data/golden/payne_zero`;
- never import the pinned checkout at reader runtime.

The chapter may show short exact source cards through the book's source helper.
Generated notebooks and HTML remain build products.

## Chapter 3 acceptance checklist

Chapter 3 is source-faithful only when:

- the two-level derivation defines level, statistical weight, stage, and
  partition function before production names appear;
- \(n\), \(n/U\), bound-level \(n_i\), and ion fraction are never conflated;
- every displayed production function is AST-equal or byte-equal to the
  pinned source;
- atmosphere and synthesis table bundles retain separate identities;
- exact/reference constants are used in the formulas that require them;
- special, packed, PFIRON, ground, lowering, and occupation paths are each
  represented without a source dump;
- the exact full closure is described as a damped fixed point, not Newton;
- the final-population staleness and subsequent atmosphere refresh are
  documented and tested;
- atom-only atmosphere closure uses real depth `prange`, while synthesis uses
  device-batched depth algebra and an ordered element loop;
- molecule-enabled ordering is deferred honestly to Chapter 4;
- charge closure and population/truncation accounting are checked under
  declared tolerances;
- `mass_density`, collision proxy, Doppler support, and atomic internal energy
  carry units and exact path-specific conventions;
- the `(D,1006)`, `(D,99,6)`, and `(D,6,139)` layouts are all shown with their
  different axis meanings;
- mode-11 and mode-12 packed schedules remain distinct;
- the fixed-\(n_e\) route preserves supplied electron density and is not
  credited with charge closure;
- data assets have one role, a source hash, array hashes, shapes, dtypes,
  units, and regeneration commands;
- goldens are loaded only after the reader computation;
- CPU, CUDA, and MPS claims use separately measured tolerance profiles;
- the summary contains no new concept and links directly to the molecular
  coupling need in Chapter 4;
- there is no detached exercise section, legacy terminology, invented API,
  universal layout, universal constant namespace, or redundant Chapter 2
  tutorial.
