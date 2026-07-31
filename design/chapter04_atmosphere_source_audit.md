# Chapter 4 atmosphere molecular-equilibrium source audit

Status: read-only source audit, not chapter prose  
Oracle repository: `/Users/ysting/payne-zero`  
Pinned commit: `9c44001feae40b85146630499e6f8a5fed42e5af`  
Audit date: 2026-07-30

This document defines the exact atmosphere-side molecular-equilibrium contract
that Chapter 4 must teach and reproduce. It deliberately separates the
scientific/numerical path from outer atmosphere orchestration. It does not
authorize edits to the oracle.

## 1. Executive corrections to the chapter plan

The atmosphere molecular solver is **not** a log-coordinate Newton solve and it
is **not** a depth-batched or `prange` solve.

- The Newton unknowns are ordinary, positive number densities in `float64`.
- One depth is solved at a time.
- The returned density vector at depth `d-1`, multiplied by
  `gas_pressure[d] / gas_pressure[d-1]`, seeds depth `d`.
- The residual and Jacobian assembly and the positivity/damping update are
  Numba-compiled, but `np.linalg.solve` / `np.linalg.lstsq` and the sequential
  depth continuation remain in Python.
- Molecular equilibrium constants are frozen during each one-depth Newton
  solve. Constants that use Saha populations are constructed from the
  pre-solve runtime electron density and charge-square density.
- The implementation tests convergence using the relative Newton step, not the
  residual norm. Hitting 200 iterations silently returns the last density
  vector.
- A separate synthesis implementation may use log-products internally. That is
  not the algorithm audited here.

These corrections are P0 for Chapter 4: teaching any other algorithm would not
rebuild the atmosphere path.

## 2. Pinned source inventory

### 2.1 Requested source files

| File | SHA-256 | Exact responsibility |
|---|---|---|
| `payne_zero_atmosphere/molecular_data.py` | `705c3072d79c8019c948ce0fa2c82052f232816d453e10a7c8e5fc5a8f5ce249` | Catalog schema, legacy fixed-width parser, base-100 species decoding |
| `payne_zero_atmosphere/molecular_equilibrium.py` | `4c9665148a57fadf4837f193c9f4247fbab44b788e7fb76ef3cae7ebdb3c3d86` | Equilibrium constants, density-space residual/Jacobian, Newton update, depth continuation, molecular populations, molecular internal energy |
| `payne_zero_atmosphere/equation_of_state.py` | `719b316327fd6fb76dfa5267a2c9022d1384c269a0fa20d5b2ec733671ae3fa2` | Molecular/atomic dispatch, once-per-temperature-index solve cache, packed population fill |
| `payne_zero_atmosphere/runner.py` | `05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7` | Live population preparation, structured diagnostic preparation, convection finite differences, outer-iteration lifecycle, output boundaries |
| `payne_zero_atmosphere/population_layout.py` | `36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0` | Packed atomic and selected molecular population schedule |
| `payne_zero_atmosphere/synthesis_bridge.py` | `142a960b5e710823754b02766803b3c1dd8c48c9945fdfabe560b4ee7e1acb50` | Live/debug packed-state conversion and release/product handoff |
| `payne_zero_atmosphere/specific_internal_energy.py` | `de06ba732ce1333d111a52223e39f5b4f80eece8cfc4ff2f30de9739e16d7ec5` | Atomic fallback specific internal energy |

### 2.2 Supporting source files needed to close the contract

| File | SHA-256 | Why it matters |
|---|---|---|
| `payne_zero_atmosphere/runtime_state.py` | `fae240ec00f6f89d7c2a7ef721ce6e6539be234e523291fd6e8a096d731430e8` | Runtime array shapes, seeds, abundance and mass conventions |
| `payne_zero_atmosphere/run_setup.py` | `de7cf08b936585dbcfa2e572c026fafa3f10282a99c27b834b62db0f3f2888c9` | `molecules_enabled` and `pressure_iteration_enabled` gates |
| `payne_zero_atmosphere/config.py` | `51e19846fb81c832ae57334faf3da2c1e4fc2ef9edf6e08467ef7296e4640b45` | Public molecular inputs and convection perturbation control |
| `payne_zero_atmosphere/source_catalogs.py` | `a9ea21735c9d4964b785d76c89c9fc976a30ed75f8b6f9d4f7c6aaa4e77dae36` | Default atmosphere molecular-catalog resolution |

## 3. Physical assets and exact identities

### 3.1 Direct atmosphere molecular assets

| Asset | SHA-256 | Active content |
|---|---|---|
| `source_data_files/source_catalogs/lines/molecular_equilibrium_atmosphere.npz` | `971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644` | 170 active species records, 23 equations, 481 component entries |
| `source_data_files/atmosphere_tables/molecular_equilibrium_tables.npz` | `1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe` | `h2_partition_function (200,) float64`; also `atomic_mass_amu (99,) float64`, which this molecular solver does not consume |

The canonical atmosphere catalog arrays are:

| Key | Shape | Dtype | Active slice |
|---|---:|---|---|
| `molecule_count` | scalar | `int64` | `170` |
| `equation_count` | scalar | `int64` | `23` |
| `component_count` | scalar | `int64` | `481` |
| `molecule_codes` | `(200,)` | `float64` | `[:170]` |
| `equilibrium_coefficients` | `(7, 200)` | `float64` | `[:, :170]`; row 6 is zero and unused |
| `component_start_indices` | `(201,)` | `int32` | `[:171]` |
| `component_equation_indices` | `(600,)` | `int32` | `[:481]` |
| `equation_species_codes` | `(35,)` | `int32` | `[:23]` |
| `species_to_equation_index` | `(102,)` | `int32` | lookup table |

The active equation-species codes are exactly

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 19, 20,
 22, 23, 24, 26, 100]
```

Index 0 is reserved for total nuclei density. Code 100 is the electron
equation. Code 101 is not an equation row: it maps to the one-past-the-end
sentinel `equation_count` and means an inverse electron factor. This semantic
is implemented by `molecular_data.py:133-151, 159-178` and by the divide branch
in `molecular_equilibrium.py:323-329, 798-805`.

Catalog branch counts that should be pinned:

- 76 active records have nonzero coefficient 0 and use a molecular
  dissociation/association expression.
- 94 active records have coefficient 0 and use identity or Saha-derived
  constants.
- Component arities 1 through 6 occur with counts
  `[18, 69, 41, 17, 16, 9]`.
- The 481 active components include 218 inverse-electron sentinel entries.
- Fourteen active records contain an ordinary electron component; in every one
  it is last, and none of those records also contains an inverse-electron
  sentinel.

### 3.2 Indirect Saha, state, and bridge assets

The molecular path calls the Chapter 3 Saha/partition implementation, so these
are transitive exact dependencies:

| Asset | SHA-256 | Relevant shape |
|---|---|---|
| `atmosphere_tables/iron_group_partition_tables.npz` | `137629dea64eca46f77ea3656c18305ade912a468d7eb27029544c0106cc3296` | `(7, 56, 10, 9) float64` |
| `atmosphere_tables/ionization_potential_tables.npz` | `82a2e82f2015da02c3d2bce77ca5337aa2b9c4e23d8d6219da07895896ca8a50` | `(999,) float64` |
| `atmosphere_tables/packed_level_metadata.npz` | `de5f17b6a9eaec1d1b07e96fd02ff014279cd8eaa9f976fefde0e2a153961bc3` | `(6, 365) int64` |
| `atmosphere_tables/special_partition_tables.npz` | `7d737524aacda1cc2281e5b18ff49f240ca34665dbe6c96d4dd0f39db4aedd22` | named H/He/C/Mg/Al/Si/Na/O/B/K levels plus `(29,)` offsets |
| `atmosphere_tables/isotope_tables.npz` | `53c8d315fb53f1e051dc2752b028fc270d7c17a2c1042279c04ffcb750aef5c6` | `major_isotope_mass_amu (1006,)`; isotope records `(10, 2, 1006)` |
| `synthesis_tables/continuum_edge_grid.npz` | `11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e` | structured-handoff continuum edge arrays |

The release handoff invokes the synthesis builder, whose molecular catalog is
different:

| Asset | SHA-256 | Counts |
|---|---|---|
| `source_catalogs/lines/molecular_equilibrium_synthesis.npz` | `3e8c1ea69fe672b9886bda38922f868c6d2ac2b43c4eb0d7750620241c238d28` | 190 species, 23 equations |

This 170-versus-190 distinction is an interface fact, not an optional detail.

## 4. Catalog encoding contract

Source: `molecular_data.py:11-18, 41-56, 59-190`.

### 4.1 Canonical and provenance inputs

- Runtime canonical input is the parsed `.npz`; it is loaded without shape or
  count validation.
- The historical fixed-width text reader remains for provenance.
- A record stores an 18-character molecule code followed by six numeric fields
  ending at column 80.
- The allocated coefficient matrix has seven rows, but the atmosphere parser
  writes only rows 0 through 5. The seventh row remains zero.
- Blank, `C`, `c`, and `#` records are skipped. Molecule code `0.0` terminates
  the text read.

### 4.2 Base-100 component code

The eight place values are

```text
1e14, 1e12, 1e10, 1e8, 1e6, 1e4, 1e2, 1
```

Each two-digit block is an atomic number. A decoded zero block is converted to
species code 100, the electron. The fractional hundredths of the molecule code
give positive ion charge; each unit of positive charge appends one code-101
inverse-electron component.

Hard catalog limits are:

- at most 200 molecule records;
- at most 600 component entries;
- at most 35 equation rows.

The equation rows are constructed from used species 1 through 100 in ascending
order. Code 101 maps to `equation_count`, deliberately one beyond the final
valid equation row.

### 4.3 Meaning for the equilibrium product

For molecule/species record `m`, let `K_m` be its equilibrium constant. Let
`x_j` be the density-space unknown for an in-range component equation, and let
`x_e` be the electron equation density. The implementation evaluates

```text
M_m = K_m
      × product(x_j for every ordinary component occurrence)
      ÷ product(x_e for every code-101 sentinel occurrence).
```

Repeated components appear repeatedly in the product and therefore encode
stoichiometric multiplicity. Code 100 is an ordinary in-range electron
component and multiplies by `x_e`; it represents a negative ion. The
negative-ion charge correction assumes that this electron component is the
last component in the record.

## 5. Runtime state, shapes, axes, and units

Sources: `runtime_state.py:16, 128-151, 192-229`;
`molecular_equilibrium.py:57-114`.

Let `D` be the number of atmosphere depth layers, ordered outermost to
innermost.

### 5.1 Atmosphere runtime arrays

| Field | Shape | Unit / meaning |
|---|---:|---|
| `gas_pressure` | `(D,)` | dyn cm^-2 = erg cm^-3 |
| `electron_density` | `(D,)` | cm^-3 |
| `total_nuclei_number_density` | `(D,)` | cm^-3 |
| `mass_density` | `(D,)` | g cm^-3 |
| `charge_square_density` | `(D,)` | cm^-3, charge-square weighted particles including electrons |
| `elemental_abundances_by_layer` | `(D, 99)` | dimensionless number fractions on the atmosphere deck scale |
| `mean_nuclear_mass_amu` | `(D,)` | amu per reference nucleus |
| `ion_stage_populations_by_packed_slot` | `(D, 1006)` | cm^-3 |
| `partition_normalized_populations_by_packed_slot` | `(D, 1006)` | cm^-3 divided by the relevant internal partition function |
| `specific_internal_energy` | `(D,)` | erg g^-1 |
| `major_isotope_mass_amu` | `(1006,)` | amu |
| `fractional_doppler_widths` | `(D, 1006)` after update | dimensionless, `Delta nu_D / nu` |

The initial runtime seed is

```text
n_particles = gas_pressure / thermal_energy_erg
n_nuclei    = n_particles - input_electron_density
rho         = n_nuclei × mean_nuclear_mass_amu × atomic_mass_gram
charge_square_density = max(2 × input_electron_density, 1e-30)
```

`update_charge_square_density` then applies the exact excess correction at
`runtime_state.py:232-250` before molecular equilibrium.

### 5.2 Molecular state arrays

For the pinned atmosphere catalog, `M=170` and `E=23`.

| Field | Shape | Lifecycle / unit |
|---|---:|---|
| `molecular_populations` | `(D, M)` | raw species number density, cm^-3 |
| `partition_normalized_molecular_populations` | `(D, M)` | partition-normalized species density, cm^-3 |
| `molecular_equation_densities` | `(D, E)` | physical equation densities in cm^-3 immediately after Newton; in normal mode column 0 remains physical total-nuclei density while columns 1 onward are overwritten by translational/partition-normalized basis values |
| `previous_molecular_equation_densities` | `(D, E)` | retained physical-density warm-start copy |

The in-place unit/lifecycle change of `molecular_equation_densities` is easy to
miss and must not be hidden behind its name.

## 6. Exact equilibrium constants

Sources: `molecular_equilibrium.py:117-156, 694-757, 1101-1147`.

All constants for one depth are computed once before Newton. They are not
recomputed inside the Newton loop.

### 6.1 H2 special branch

For molecule code `101.0`, and only when `T <= 20000 K`,

```text
K_H2(T) =
    U_H2(T) × 2^(3/2) / 4
    ------------------------------------------------
    [2 pi × 1.008 m_u × k_B T / h^2]^(3/2)
    × exp[36118.11 × h c / (k_B T)].
```

`U_H2` comes from the 200-value H2 table. The interpolation:

- clamps nonfinite or `T <= 100 K` to 100 K;
- clamps `T >= 19900 K` to 19900 K;
- uses `index = min(199, max(1, int(T/100)))`;
- interpolates table entries `index-1` and `index` using
  `(T-index*100)/100`.

At and above the 19900 K clamp, this exact indexing evaluates the lower of
those two entries at a zero interpolation fraction. Chapter code must reproduce
the source, not substitute `np.interp` without first proving identical endpoint
behavior.

### 6.2 Non-H2 coefficient branch

For nonzero first coefficient `c1`, non-H2 constants are exactly zero above
10000 K. At or below 10000 K,

```text
K_m(T) = exp[
    c1 / (T / 11604.5)
    - c2
    + c3 T - c4 T^2 + c5 T^3 - c6 T^4
    - 1.5 (C_m - 2 q_m - 1) ln T
].
```

Here `C_m` is the number of catalog components and `q_m` is the positive ion
charge encoded in the fractional molecule code. `c1` is therefore used as an
energy in eV. The remaining coefficient units are whatever makes the exponent
dimensionless.

There is no overflow guard around this `exp`.

### 6.3 Zero-first-coefficient branch

- If component count is one, `K_m = 1`.
- If component count exceeds one, the implementation calls the exact
  atmosphere Saha kernel in population mode 12 and forms

```text
K_m = f_last / f_first × max(n_e_runtime, 1e-300)^(C_m-1).
```

The Saha call uses the runtime electron density and the runtime
`max(charge_square_density, 1e-30)`. Because continuum lowering enters the Saha
partitions, the cancellation with the explicit electron-density power is not
strictly density-independent.

## 7. Density-space Newton system

Sources: `molecular_equilibrium.py:760-885, 1150-1209`.

### 7.1 Unknown vector

For `E=equation_count`, the vector is:

```text
x[0]       total nuclei number density
x[1:E-1]   free elemental basis densities in equation_species_codes order
x[E-1]     electron density, when the final species code is 100
```

Every entry is an ordinary `float64` density. No logarithm is taken.

### 7.2 Residual

For active records with more than one component, define `M_m(x)` by the
catalog product in Section 4.3. The source assembles:

```text
R_0 = -P_gas/(k_B T) + sum_i=1^(E-1) x_i + sum_m M_m

R_a = x_a - A_a x_0 + sum_m nu_ma M_m

R_e = -x_e + sum_m q_m M_m.
```

`A_a` is the layer abundance floored to `1e-20` for species codes 1 through
99. `nu_ma` is component multiplicity. `q_m` is positive for inverse-electron
sentinels and negative for ordinary code-100 electron components. The source
implements negative charge by first adding the generic component contribution
and then subtracting it twice when the record's last component is the electron
equation.

Records of component count one are not added as `M_m` terms in the Newton
residual. Their free basis density is already present in `sum x_i`.

The pressure row is particle conservation. The elemental rows are nuclei
conservation. The final row is charge conservation.

### 7.3 Analytic Jacobian

For each ordinary component occurrence,

```text
dM_m/dx_j = +M_m / max(x_j, 1e-300).
```

For each inverse-electron sentinel,

```text
dM_m/dx_e = -M_m / max(x_e, 1e-300).
```

The implementation loops over occurrences, so repeated components naturally
sum to their multiplicity. It inserts these derivatives into the pressure row
and every affected conservation row. The initial diagonal terms are
`J[a,a]=1`, `J[a,0]=-A_a`, and `J[e,e]=-1`.

### 7.4 Linear solve and fallback

Each iteration executes:

```text
delta = np.linalg.solve(J, R)
```

and falls back only on `np.linalg.LinAlgError`:

```text
delta = np.linalg.lstsq(J, R, rcond=None)[0].
```

The returned vector size is checked. Finiteness, conditioning, and residual
decrease are not checked.

### 7.5 Damping, positivity, and convergence

The exact update order for each equation index is:

1. Mark the solve as still iterating when
   `abs(delta_i) / max(abs(x_i), 1e-300) > 1e-4`.
2. If `previous_delta_i * delta_i < 0`, multiply `delta_i` by `0.69`.
3. Form `updated = x_i - delta_i` and `floor = x_i / 100`.
4. If `abs(updated) >= floor`, assign `x_i = abs(updated)`.
5. Otherwise assign `x_i = x_i / scale`, with `scale` initially 100 for the
   whole vector update.
6. If the small-update fallback coincides with a sign change, replace the
   shared `scale` by `sqrt(scale)` for subsequent vector entries.
7. Store the possibly damped delta as `previous_delta_i`.

Consequences:

- positivity is enforced by reflection through `abs`, not by a log transform;
- the small-density fallback is order-dependent because `scale` is shared
  across equation indices;
- convergence is based on the pre-update relative step;
- the final below-tolerance update is still applied;
- after 200 iterations, the last vector is returned without a warning or
  convergence flag.

## 8. Seeds and sequential depth continuation

Source: `molecular_equilibrium.py:244-340`.

For the first layer,

```text
n_particle,0 = P_0 / (k_B T_0)
x_0 = n_particle,0 / 2
if T_0 < 4000 K: x_0 = n_particle,0
n_e,seed = x_0 / 10
x_a = n_e,seed × A_a
x_e = n_e,seed
runtime electron_density[0] = n_e,seed.
```

For each later layer,

```text
ratio = gas_pressure[d] / max(gas_pressure[d-1], 1e-300)
equation_density *= ratio
runtime electron_density[d] =
    runtime electron_density[d-1] × ratio.
```

Then the one-depth Newton solve runs. Afterward:

```text
total_nuclei_number_density[d] = x[0]
mass_density[d] = x[0] × mean_nuclear_mass_amu[d] × atomic_mass_gram
electron_density[d] = x[e].
```

Important lifecycle boundaries:

- Continuation exists across depths inside one call.
- A new `MolecularEquilibriumState` is allocated for every outer atmosphere
  iteration, so the physical equation-density continuation is not carried
  across outer iterations.
- In specific-internal-energy mode, a nonzero saved row replaces the
  pressure-scaled equation-density seed at that depth.
- The runtime electron-density seed used to build Saha-derived constants is
  still the pressure-scaled runtime value; constants remain frozen during the
  solve.
- `charge_square_density` is seeded before the molecular solve and is not
  recomputed from the final molecular/ionic state.

The last two bullets are exact implementation behavior and require explicit
parity tests.

## 9. Raw and partition-normalized populations

Sources: `molecular_equilibrium.py:315-353, 356-485, 488-593`.

### 9.1 Raw populations

After Newton, constants are recomputed once using the now-updated runtime
electron density. Every active catalog record, including one-component
records, is evaluated:

```text
molecular_populations[d,m] = K_m × component density product.
```

This second constant evaluation can differ slightly from the constants frozen
inside Newton when density-dependent partition lowering matters.

### 9.2 Saved physical seed

In normal mode, the physical `(D,E)` equation-density array is copied to
`previous_molecular_equation_densities` before normalization. This is the
warm-start state used by convection perturbations.

### 9.3 Equation-basis normalization

Unless `population_mode` is 2 or 12, the source modifies columns 1 onward of
`molecular_equation_densities` in place. Column 0 remains the physical total
nuclei number density.

For an elemental basis species:

```text
x_a <- x_a /
       [U_a(T) × 1.8786e20 × sqrt((atomic_mass_amu × T)^3)].
```

`U_a` is the first value from Saha population mode 3.

For the electron equation:

```text
x_e <- x_e / [2 × 2.4148e15 × T × sqrt(T)].
```

These transformed basis values are dimensionless translational/partition
ratios, not physical cm^-3 densities.

### 9.4 Partition-normalized molecular species

For a nonzero-coefficient molecule, the source multiplies the transformed
component product by

```text
exp(c1 / (T/11604.5))
× 1.8786e20 × sqrt((molecule_mass_amu × T)^3).
```

The molecule mass is the sum of ordinary atomic components; electron and
inverse-electron sentinels add no mass.

For a zero-first-coefficient species, the source instead uses

```text
raw_molecular_population / first_mode3_partition_value.
```

The result is stored in
`partition_normalized_molecular_populations (D,M)`.

### 9.5 Population-mode dispatch

- Codes `>=100` are looked up by molecule code to `1e-3` tolerance.
- Modes 1 and 11 return the partition-normalized molecular population.
- Other modes return the raw molecular population.
- For codes `<100`, the molecular catalog is tried first, ion stage by ion
  stage. If no family exists, the code falls back to the atomic Saha path.
- A present element family with a missing exact stage writes zero rather than
  falling back.
- The atomic fallback calls Saha with `charge_square_density=None`, then scales
  by total nuclei density and abundance.

## 10. Packed population schedule

Source: `population_layout.py:72-204`;
dispatch: `equation_of_state.py:1550-1676`.

- Packed width is 1006.
- The ordinary atomic schedule has 198 jobs.
- Enabling molecules appends 32 jobs: modes 1 and 11 for each of 16 selected
  molecular codes.
- Total molecule-enabled schedule length is 230.

The exact selected molecule destinations are:

| Code | One-based packed slot |
|---:|---:|
| 101.00 | 841 |
| 106.00 | 846 |
| 107.00 | 847 |
| 108.00 | 848 |
| 112.00 | 851 |
| 114.00 | 853 |
| 120.00 | 858 |
| 124.00 | 862 |
| 126.00 | 864 |
| 606.00 | 868 |
| 607.00 | 869 |
| 608.00 | 870 |
| 814.00 | 889 |
| 822.00 | 895 |
| 823.00 | 896 |
| 10108.00 | 940 |

Both molecular modes write the
`partition_normalized_populations_by_packed_slot` target, and for a code
`>=100` both retrieve the same partition-normalized source. This is a real
legacy schedule duplication; it should be identified as such rather than
inventing two physical quantities.

The full `(D,170)` molecular arrays remain authoritative for all catalog
species. Only 16 are duplicated into selected packed opacity slots.

## 11. Once-per-temperature-index orchestration

Source: `equation_of_state.py:1550-1676`.

`temperature_iteration_cache["pops_itemp"]` controls whether the expensive EOS
solve runs.

- Molecular branch: when pressure iteration is enabled and the requested
  temperature index differs from the cache, `solve_molecular_equilibrium`
  runs, then the cache is updated.
- Atom-only branch: the corresponding event calls
  `iterate_electron_density`.
- The runner deliberately calls `populate_species(code=0, mode=1, ...)` first.
  Code zero triggers the once-only solve and then returns without filling a
  species output.
- `populate_all_species` subsequently walks the 230-job schedule at the same
  temperature index, so it consumes the solved state without resolving.
- Convection finite differences use four distinct temperature indices to force
  one solve for each `T+`, `T-`, `P+`, and `P-` state.

This cache choreography belongs in a compact call-flow diagram or small
instrumented example. The full runner call graph does not belong in a teaching
cell.

## 12. Numba and parallelism contract

Source: `molecular_equilibrium.py:17-30, 623-629, 631-1025`.

- Numba is a hard dependency. Import failure raises immediately.
- The four molecular kernels use `numba.njit(cache=True, nogil=True)`.
- They are:
  1. equilibrium constants;
  2. residual/Jacobian assembly;
  3. Newton damping/positivity update;
  4. molecular internal-energy accumulation.
- None uses `parallel=True` or `numba.prange`.
- The depth loop is sequential because each depth consumes the previous
  pressure-scaled solution.
- `np.linalg.solve` and `np.linalg.lstsq` remain outside the Numba kernels.
- Depth-batched Saha calls are used when the same element/stage/mode is needed
  at every depth, but that does not make the molecular Newton solve
  depth-parallel.
- The atom-only electron-density path in `equation_of_state.py:936-1052,
  1393-1500` does use a `parallel=True` `prange` depth sweep. Chapter 4 must
  contrast this independence with molecular continuation rather than imply
  that all atmosphere EOS work parallelizes the same way.

No GPU path exists in this atmosphere molecular solver.

## 13. Specific internal-energy lifecycle

Sources: `molecular_equilibrium.py:332-353, 596-620, 888-1025, 1212-1366`;
`runner.py:374-569`; atomic fallback in `specific_internal_energy.py:25-171`.

### 13.1 Normal molecular solve

After normal population filling, runtime specific internal energy is set to

```text
e_specific = 1.5 × gas_pressure / mass_density
```

in erg g^-1. This is the translational ideal-particle value. The molecular
dissociation/partition contribution is not inserted at this stage.

If `population_mode` is 2 or 12, the function returns before partition
normalization and before assigning this value. The operational runner's
zero-code priming call uses mode 1.

### 13.2 Convection finite-difference mode

Before four finite-difference solves:

1. Save all original atmosphere/runtime/molecular arrays.
2. Copy `previous_molecular_equation_densities`.
3. Set `specific_internal_energy_mode_enabled=True`.

For each perturbation:

- the distinct temperature index forces a molecular solve;
- any nonzero saved equation-density row becomes that depth's Newton seed;
- `previous_molecular_equation_densities` is not overwritten;
- partition-normalized molecular populations are not refilled;
- runtime specific internal energy is replaced by the full molecular energy.

The four perturbations are exactly `T×1.001`, `T×0.999`, `P×1.001`, and
`P×0.999`.

### 13.3 Molecular energy expression

For each layer, the energy-density accumulator starts at

```text
1.5 × (P_gas / thermal_energy_erg) × thermal_energy_erg
= 1.5 P_gas.
```

Each positive-density molecular species adds a dissociation/ionization and
partition-derivative contribution. Partition derivatives use

```text
(Q(T×1.001) - Q(T×0.999))
-------------------------------- × 1000,
max(Q(T×1.001) + Q(T×0.999), 1e-30)
```

which is the exact centered relative derivative convention in the source.

- H2 uses the tabulated partition and dissociation wavenumber `36118.11 cm^-1`.
- Other nonzero-coefficient molecules use the coefficient polynomial,
  relevant mode-3 atomic partitions, and `c1/(T/11604.5)`.
- Zero-first-coefficient atomic/ionic records use mode-5 Saha partition and
  ionization outputs.

The final energy density is divided by
`max(mass_density, 1e-300)` to produce erg g^-1.

### 13.4 Thermal-energy reference and restoration

`run_atmosphere_model` copies the initial setup atmosphere's
`thermal_energy_erg` once and passes that reference into each newly allocated
molecular state. Normal equilibrium constants use the current temperature,
not this stored thermal array. The stored array matters in molecular
specific-energy mode.

The public config default
`molecular_convection_thermal_tracks_perturbation=True` causes the runner to
replace the stored molecular thermal energy with each perturbed atmosphere's
thermal energy before its solve. The lower-level finite-difference function's
own default is `False`; the runner explicitly supplies the config value.

The `finally` block restores every saved array and cache. If any entry of the
original runtime specific-energy vector was nonzero, it restores the whole
vector. Otherwise it computes the atomic fallback.

### 13.5 Atomic fallback boundary

`compute_atomic_specific_internal_energy`:

- uses 840 packed atomic slots;
- constructs cumulative ionization potentials from the 999-value table;
- evaluates mode-13 partitions at `T×1.001` and `T×0.999`;
- adds translational, ionization, and partition-derivative energy;
- divides by mass density.

This is a fallback/restoration boundary, not the main molecular-energy
algorithm. Re-teaching all Chapter 3 partition internals here would be
redundant.

## 14. Live, diagnostic, and product handoff boundaries

### 14.1 Live atmosphere population state

Source: `runner.py:200-285`.

`prepare_population_state`:

1. resolves setup;
2. builds a fresh runtime state from the current atmosphere;
3. seeds charge-square density;
4. loads the 170-species atmosphere molecular catalog when enabled;
5. allocates a molecular state;
6. when pressure iteration is enabled, primes the molecular solve with code
   zero and fills the complete packed schedule;
7. computes Doppler and line-strength support arrays.

If `pressure_iteration_enabled=False`, this function does not fill either
atomic or molecular population arrays before the Doppler update. The public
setup can reach this branch through atmosphere metadata.

### 14.2 Structured diagnostic population rebuild

Source: `runner.py:288-371`.

The docstring says “at fixed final electron density.” The actual branches are:

- atom-only: no electron-density iteration; populate atomic Saha arrays at the
  input/final electron density;
- molecules enabled: unconditionally call the code-zero molecular solve with
  `pressure_iteration_enabled=True`, which replaces runtime electron density,
  total nuclei density, and mass density before the fixed-fill schedule.

Therefore “fixed final electron density” is true only for the atom-only branch.

### 14.3 Live/debug structured bridge

Sources: `synthesis_bridge.py:226-570, 573-612, 615-685`;
debug save in `runner.py:1534-1560`.

The live packed bridge:

- requires at least 351 packed slots;
- converts packed atomic arrays to `(D, 6, 139)`;
- places ordinary atomic species in columns 0 through 98;
- maps 54 named molecular codes into the sixth stage axis at pseudo-species
  columns `species_code//6 - 1`;
- compares molecule codes through
  `int(round(molecule_code * 100))` and sums duplicate matching columns;
- uses a 139-column axis because the largest pseudo-species mapping reaches
  column 138;
- exports H, He, C, Mg, Al, Si, Fe sentinel arrays;
- exports raw H2, exact molecular H2 when available, or a fallback H2 estimate;
- adds schema version 4 and continuum-edge arrays.

The raw structured fields and units are:

| Field | Shape | Unit |
|---|---:|---|
| `temperature` | `(D,)` | K |
| `gas_pressure` | `(D,)` | dyn cm^-2 |
| `electron_density` | `(D,)` | cm^-3 |
| `mass_density` | `(D,)` | g cm^-3 |
| `column_mass` | `(D,)` | g cm^-2 |
| `ion_stage_populations` | `(D,6,139)` | cm^-3 |
| `partition_normalized_populations` | `(D,6,139)` | cm^-3 / internal partition |
| `fractional_doppler_widths` | `(D,6,139)` | dimensionless |
| `hc_over_kt` | `(D,)` | cm |
| `microturbulence` | `(D,)` | cm s^-1 |
| `elemental_abundances` | `(99,)` | dimensionless; bridge rejects layer-dependent differences |

The continuum-edge arrays have exact shapes `(341,)`, `(341,)`, `(341,)`,
`(1020,)`, `(340,)`, and `(340,)` for signed frequency, wavelength,
wavenumber, sampled frequency, midpoint wavelength, and squared interval
weight respectively.

The fallback H2 bridge value is

```text
n_H2 = n_HI^2 × exp[
    4.478 / (8.617333262e-5 T)
    - 46.4584
    + 1.63660e-3 T
    - 4.93992e-7 T^2
    + 1.11822e-10 T^3
    - 1.49567e-14 T^4
    + 1.06206e-18 T^5
    - 3.08720e-23 T^6
    - 1.5 ln T
].
```

It is set to zero for `T > 9000 K`. This fallback is used only when the entire
catalog H2 vector has no positive entry; it is not applied selectively to zero
depths in an otherwise positive catalog vector.

### 14.4 Release/product handoff

Source: `synthesis_bridge.py:701-736`; call site `runner.py:1966-1995`.

The release handoff does **not** serialize the live atmosphere molecular state.
After convergence it passes only final atmosphere columns and abundances to
`payne_zero_synthesis.build_structured_atmosphere`, which rebuilds the
structured product through the synthesis implementation and its 190-species
catalog. The runner records the source as
`final_fixed_column_quantized_arrays`.

Accordingly:

- live/debug bridge parity and product/release parity are different tests;
- atmosphere 170-species populations are not the release product's direct
  population source;
- Chapter 4 should stop at the explicit handoff contract and leave the
  synthesis solver's internals to its owning chapter.

## 15. Teach inline versus stage as production orchestration

### 15.1 Teach and execute inline

The following are scientifically essential and can be kept bite-sized:

1. Decode three representative base-100 codes: neutral molecule, positive ion
   with inverse electron, negative ion with ordinary electron.
2. Load and validate the pinned catalog counts and active slices.
3. Plot one H2 equilibrium constant against temperature, explicitly showing
   the 19900 K partition clamp and 20000 K activity gate.
4. Compare one polynomial molecule below/above the 10000 K gate.
5. Build a minimal H/H+/H2 pressure, nuclei, and charge residual in readable
   NumPy.
6. Verify the analytic Jacobian against a finite-difference Jacobian.
7. Demonstrate all three Newton update branches: normal step, sign-change
   damping, and positivity/small-value fallback.
8. Solve one layer and print the relative-step history.
9. Solve a short ordered depth column and compare pressure-scaled continuation
   with cold restarts.
10. Show raw versus partition-normalized molecular populations and their units.
11. Demonstrate the temperature-index cache with one counted solve.
12. Show why the molecular depth loop cannot use `prange`, while the atom-only
   Chapter 3 loop can.
13. Demonstrate saved equation-density warm starts for `T+`, `T-`, `P+`, `P-`.
14. Validate live handoff axes and sentinel fields on a sliced active catalog.
15. Compare a chapter implementation against a pinned atmosphere golden over
   every runtime and molecular field.

### 15.2 Keep staged, but expose the boundary

These are exact production details that should be named and tested without
dumping long source blocks into the notebook:

- all 170 catalog records and 481 component entries;
- all 230 packed population jobs;
- the full Chapter 3 Saha/partition table machinery;
- Numba cache repacking and kernel-cache attachment to the catalog object;
- outer opacity, transfer, temperature-correction, and remap orchestration;
- 54-entry molecular pseudo-species bridge map;
- source-catalog environment context for the independent synthesis builder;
- debug and diagnostics file writing.

The staged code must retain exact readable names and be inspectable from the
chapter. The prose should explain what crosses each boundary and why.

## 16. Required parity and acceptance tests

### 16.1 Source/data identity

- Assert the pinned commit and every direct asset SHA above.
- Assert all catalog scalar counts, shapes, dtypes, active species codes, and
  zero padding.
- Assert coefficient row 6 is zero for the atmosphere catalog.
- Assert the atmosphere/synthesis catalog count distinction `170 != 190`.

### 16.2 Catalog semantics

- Exact fixed-width parser record with six coefficients.
- Base-100 decode for repeated atoms, code-100 electron, and code-101 inverse
  electron.
- Positive-ion and negative-ion component order/sign tests.
- Hard-limit failures for molecules, components, and equations.
- Canonical `.npz` versus provenance-text active-array equality.

### 16.3 Equilibrium constants

- H2 values at 100, 101, 19899, 19900, 20000, and just above 20000 K.
- Non-H2 polynomial values at 10000 K and just above it.
- Zero-coefficient one-component identity.
- Saha-derived ion constant with exact Chapter 3 density-lowering inputs.
- Explicit test that constants are frozen during Newton.
- Explicit test that post-solve population constants are recomputed with final
  runtime electron density.

### 16.4 Residual, Jacobian, and Newton update

- Hand-computable neutral, positive-ion, and negative-ion residuals.
- Analytic versus finite-difference Jacobian over every column.
- Duplicate-component multiplicity.
- `np.linalg.solve` branch and forced `lstsq` fallback.
- Relative-step tolerance exactly `1e-4`.
- Sign-change factor exactly `0.69`.
- Reflection, `x/100` floor, and shared-scale order dependence.
- Below-tolerance final update still applied.
- 200-iteration exhaustion returns without raising.
- Nonfinite behavior is characterized rather than silently sanitized.

### 16.5 Depth and lifecycle

- First-layer seeds on each side of 4000 K.
- Pressure-ratio continuation exact array equality.
- Sequential depth order sensitivity.
- No `prange` in the molecular solve; thread-count invariance of compiled
  molecular kernels.
- Saved physical equation densities remain physical while the public working
  array is normalized in place.
- `charge_square_density` remains the pre-solve seed.
- Outer atmosphere iteration allocates a fresh molecular state.

### 16.6 Populations and packed layout

- Raw molecular populations for all 170 active records.
- Partition-normalized molecular populations for all 170 active records.
- Every selected code/slot pair in the 32-job molecular schedule.
- Demonstrate that molecular modes 1 and 11 write the same selected value.
- Present-family/missing-stage zero versus absent-family Saha fallback.
- Every packed field shape `(D,1006)` and exact zero padding.

### 16.7 Specific internal energy

- Normal `1.5 P/rho` assignment.
- Mode 2/12 early return.
- Four perturbation factors and distinct cache indices.
- Warm-start save/restore and no overwrite in energy mode.
- H2, non-H2 coefficient, and zero-coefficient energy branches.
- Default runner behavior with molecular thermal energy tracking enabled.
- Full restoration after success and after an injected exception.
- Atomic fallback only when the original whole vector is zero.

### 16.8 Handoff

- Atom-only structured diagnostic preserves input electron density.
- Molecule-enabled structured diagnostic is tested for its actual re-solve.
- Active-code-sliced live bridge produces schema-4 `(D,6,139)` arrays.
- All H/He/C/Mg/Al/Si/Fe sentinel fields equal their source packed slots.
- Every one of the 54 molecular pseudo-species mappings.
- H2 exact-source branch and all-zero fallback branch.
- Layer-dependent abundance rejection.
- Product handoff compared against the independent synthesis builder and a
  final spectrum golden; do not substitute live-bridge equality.

## 17. P0 ambiguities and blockers

### P0-1 — “Fixed final electron density” contradicts the molecule branch

- Claim: `runner.py:295`.
- Actual behavior: `runner.py:306-341` invokes the full molecular solve with
  pressure iteration enabled, replacing electron density.
- Atom-only behavior: `runner.py:343-352` does retain the input electron
  density.

**Required resolution:** Chapter 4 must describe branch-specific behavior.
Before accepting a “fixed molecular handoff” exercise or golden, decide whether
parity means preserving this exact re-solve or a future oracle correction.

### P0-2 — Live/debug molecular structured bridge has an active/padded length mismatch

- Catalog codes passed by `structured_atmosphere_from_runtime_state`:
  `synthesis_bridge.py:585-593` uses the full padded `(200,)` code array.
- Molecular populations passed beside it have shape `(D,170)`.
- Validation at `synthesis_bridge.py:472-490` requires
  `(D, codes.size)`, hence `(D,200)`.
- Debug NPZ similarly writes full padded codes and active populations at
  `runner.py:1534-1560`.

**Consequence:** the molecular live/debug wrapper cannot satisfy its own shape
contract with the pinned canonical catalog. The runner catches the resulting
diagnostic failure, but a textbook implementation must not pretend this path
currently succeeds.

**Required resolution:** For teaching, slice `molecule_codes[:molecule_count]`
only in a clearly labeled chapter adapter/golden probe, without claiming that
the pinned wrapper already does so. Product/release parity must remain tied to
the independent synthesis builder.

### P0-3 — Release product is not a serialization of atmosphere molecular state

- Runtime diagnostic exporter: `runner.py:1922-1964`.
- Product exporter: `runner.py:1966-1995`.
- Independent synthesis rebuild: `synthesis_bridge.py:701-735`.
- Catalog counts differ: atmosphere 170, synthesis 190.

**Required resolution:** Define two separate acceptance gates:

1. atmosphere molecular-state parity against an atmosphere golden;
2. release structured/spectrum parity against the synthesis product path.

Do not require byte equality between the live atmosphere bridge and product
builder outputs.

### P0-4 — Newton exhaustion is silently accepted

- Limits: `molecular_equilibrium.py:48-49`.
- Return after exhausted loop: `molecular_equilibrium.py:1180-1209`.

**Required resolution:** Exact rebuild code must preserve this behavior for
parity, while the chapter must label it as an implementation limitation.
Diagnostics may observe it externally but must not alter the result used for
the exact golden.

## 18. P1 ambiguities and audit risks

### P1-1 — Frozen Saha-derived constants use pre-solve runtime state

Constants are computed before Newton at
`molecular_equilibrium.py:1167-1171`; runtime electron density is replaced only
after Newton at `molecular_equilibrium.py:304-313`. Charge-square density is
not refreshed. Post-solve populations recompute constants at
`molecular_equilibrium.py:315-330`.

This must be taught as a fixed-point approximation inside each layer, not as a
fully coupled update of pressure-lowered partition functions.

### P1-2 — `molecular_equation_densities` changes meaning in place

Physical densities are written at `molecular_equilibrium.py:301-303`, saved at
332-335, then normalized in place at 356-415. Debug output exposes the mutated
array under the density name at `runner.py:1558-1560`.

Chapter tables and plots must label whether they use the saved physical seed or
the transformed working array.

### P1-3 — Negative-ion charge handling depends on component ordering

The negative-ion correction triggers only when the last component maps to
species code 100: `molecular_equilibrium.py:831-857`. Catalog provenance tests
must assert this ordering invariant for every negative-ion record.

### P1-4 — H2 interpolation endpoint is non-obvious

The clamp/index formula at `molecular_equilibrium.py:117-126, 642-651` is not
equivalent to an unqualified generic interpolation call. The exact 19900 K
behavior needs a pinned value test.

### P1-5 — `prepare_population_state` leaves populations empty when pressure iteration is off

The only fill is inside `if setup.pressure_iteration_enabled` at
`runner.py:242-266`; Doppler support is still computed afterward at 268-274.
The setup flag can be disabled through atmosphere metadata at
`run_setup.py:174-178`.

The chapter should identify this as a boundary/limitation. It should not imply
that this live function provides a fixed-density population fill; that role is
attempted by the separate structured diagnostic preparer.

### P1-6 — Molecular energy has two thermal-energy defaults

- Public config default is tracking enabled: `config.py:51`.
- Lower-level finalization and finite-difference defaults are disabled:
  `runner.py:384, 985`.
- The product runner explicitly forwards the public setting:
  `runner.py:1711-1713`.

Examples should enter through the product runner or pass the flag explicitly.

### P1-7 — The canonical `.npz` loader trusts its schema

`molecular_data.py:71-87` returns arrays without validating active counts,
shapes, dtypes, padding, species ordering, or component bounds. The textbook
loader may add assertions for pedagogy, but the exact production call must
still consume the same arrays and values.

### P1-8 — Population modes 1 and 11 are duplicated for selected molecules

The two schedule jobs are distinct at `population_layout.py:196-203`, but
`populate_molecular_species` selects the same normalized source for both at
`molecular_equilibrium.py:505-515`. Treat this as a verified redundant
production schedule entry, not two different molecular observables.

### P1-9 — H2 bridge fallback is global, not depth-wise

`synthesis_bridge.py:350-357` returns the catalog H2 vector when **any** depth
is positive. It applies the analytic fallback only when no depth is positive.
Mixed positive/zero depth vectors therefore retain zeros rather than filling
them depth by depth.

## 19. Chapter 4 acceptance statement

Chapter 4 is technically acceptable only when:

- it uses the density-space, sequential continuation algorithm above;
- every visible equation maps to the exact source residual and update;
- all 170 atmosphere species and 23 equations are covered by staged code and
  golden tests, even if only representative species are expanded inline;
- it distinguishes frozen pre-solve constants from post-solve population
  constants;
- it exposes the transformed `molecular_equation_densities` lifecycle;
- it demonstrates Numba acceleration without claiming molecular `prange` or
  GPU execution;
- it separates atmosphere-state parity, diagnostic bridge behavior, and
  release synthesis parity;
- it closes or explicitly labels every P0/P1 item above;
- it ends with a precise handoff: the next chapter may consume structured
  atmosphere fields, but synthesis-side molecular equilibrium owns its own
  catalog and implementation.
