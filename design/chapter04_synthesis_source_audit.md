# Chapter 4 synthesis molecular-equilibrium source audit

Status: pinned-source audit for planning and implementation; this is not chapter prose.

Pinned oracle:

- repository: `/Users/ysting/payne-zero`
- commit: `9c44001feae40b85146630499e6f8a5fed42e5af`
- audit method: source was read from the pinned Git object with `git show`; the Payne Zero checkout was not modified
- scope: the synthesis molecular-equilibrium catalog and solver, the full and fixed-electron-density EOS routes, molecular population packing, and both atmosphere-to-synthesis bridge paths

The central source result is unambiguous:

> The synthesis molecular-equilibrium solver uses **number densities as its Newton unknowns**. It evaluates the mass-action products in log space, but it does **not** solve for logarithmic unknowns. Depths are solved sequentially with pressure-scaled continuation. `jacrev` is called for one depth on every Newton iteration. `vmap` is used only after every ordered depth loop has terminated, whether by convergence or iteration exhaustion, to evaluate the final molecular populations.

Any Chapter 4 plan or prose that says “log-space Newton,” “batched chemistry solve,” or “`vmap` over the Newton solve” is source-inaccurate.

## 1. Exact source inventory

Primary synthesis sources:

| Responsibility | Pinned source location |
|---|---|
| Catalog buffers and parser | `payne_zero_synthesis/molecular_equilibrium.py:25-178` |
| Polynomial formation constants | `payne_zero_synthesis/molecular_equilibrium.py:181-281` |
| Active molecular structure | `payne_zero_synthesis/molecular_equilibrium.py:284-353` |
| Log-product boundary and residual | `payne_zero_synthesis/molecular_equilibrium.py:356-419` |
| Scaled Newton step | `payne_zero_synthesis/molecular_equilibrium.py:422-433` |
| Final molecular-density evaluation | `payne_zero_synthesis/molecular_equilibrium.py:436-457` |
| Sequential molecular solve | `payne_zero_synthesis/molecular_equilibrium.py:460-670` |
| Molecular line-population conversion | `payne_zero_synthesis/molecular_equilibrium.py:779-1008` |
| Molecular line species-code map | `payne_zero_synthesis/molecular_equilibrium.py:1011-1114` |
| Metadata and atomic-ion formation rows | `payne_zero_synthesis/molecular_equilibrium.py:1117-1206` |
| Atomic derived state | `payne_zero_synthesis/equation_of_state.py:1293-1337` |
| Molecular electron-density seed | `payne_zero_synthesis/equation_of_state.py:1340-1396` |
| Atomic/full electron-density route | `payne_zero_synthesis/equation_of_state.py:1467-1653` |
| Public `PopulationState` | `payne_zero_synthesis/equation_of_state.py:1659-1680` |
| Molecule-backed packed populations | `payne_zero_synthesis/equation_of_state.py:1780-2018` |
| Packed-stage to public-cube map | `payne_zero_synthesis/equation_of_state.py:2021-2042` |
| Full population-state API | `payne_zero_synthesis/equation_of_state.py:2045-2080` |
| Fixed-electron-density API | `payne_zero_synthesis/equation_of_state.py:2083-2147` |
| Final state assembly | `payne_zero_synthesis/equation_of_state.py:2150-2308` |
| Public atmosphere-builder boundary | `payne_zero_synthesis/synthesis.py:39-87` |
| Synthesis population bridge | `payne_zero_synthesis/pipeline.py:308-635` |
| Runtime dtype/device policy | `payne_zero_synthesis/device.py:13-56` |
| Structured-atmosphere schema | `payne_zero_synthesis/atmosphere.py:16-57` and `payne_zero_synthesis/atmosphere_schema.json` |

Atmosphere-side bridge sources relevant to the same handoff:

| Responsibility | Pinned source location |
|---|---|
| Packed atomic cube | `payne_zero_atmosphere/synthesis_bridge.py:226-282` |
| Molecule-code population lookup | `payne_zero_atmosphere/synthesis_bridge.py:285-339` |
| H2 population selection/fallback | `payne_zero_atmosphere/synthesis_bridge.py:342-389` |
| Packed-state structured mapping | `payne_zero_atmosphere/synthesis_bridge.py:392-570` |
| Live runtime-state mapping | `payne_zero_atmosphere/synthesis_bridge.py:573-612` |

Constants that must not be silently unified:

- the molecular solver imports the exact synthesis constants
  `BOLTZMANN_ERG_PER_K = 1.380649e-16` and
  `BOLTZMANN_EV_PER_K = 8.617333262e-5`;
- the atomic EOS uses the parity-pinned rounded constants
  `REFERENCE_BOLTZMANN_ERG_PER_K = 1.38054e-16`,
  `REFERENCE_BOLTZMANN_EV_PER_K = 8.6171e-5`,
  `REFERENCE_SAHA_COEFFICIENT = 2.4148e15`, and
  `REFERENCE_ATOMIC_MASS_GRAM = 1.660e-24`.

These values and the deliberate two-tier policy are defined at
`payne_zero_synthesis/constants.py:22-43`.

## 2. Catalog contract

### 2.1 Active runtime asset

The equilibrium solver resolves its default table through
`source_catalog_path("lines", "molecular_equilibrium_synthesis.npz")`
(`molecular_equilibrium.py:43-46`). Path resolution is:

1. `PAYNE_ZERO_SYNTHESIS_SOURCE_CATALOG_ROOT`;
2. `PAYNE_ZERO_SOURCE_CATALOG_ROOT`;
3. `<data_root>/source_catalogs`;

where `PAYNE_ZERO_DATA_ROOT` can replace the shared data root
(`payne_zero_synthesis/paths.py:23-39,53-80`).

Pinned LFS object:

- path: `source_data_files/source_catalogs/lines/molecular_equilibrium_synthesis.npz`
- SHA-256: `3e8c1ea69fe672b9886bda38922f868c6d2ac2b43c4eb0d7750620241c238d28`
- bytes: `18060`

Exact archive fields:

| Field | Stored shape | dtype | Active extent |
|---|---:|---|---:|
| `molecule_count` | scalar | `int64` | `190` |
| `equation_count` | scalar | `int64` | `23` |
| `molecule_codes` | `(200,)` | `float64` | first `190` |
| `equilibrium_coefficients` | `(7, 200)` | `float64` | first `190` molecule columns |
| `component_start_indices` | `(201,)` | `int32` | first `191` |
| `component_equation_indices` | `(600,)` | `int32` | first `548` |
| `equation_species_codes` | `(30,)` | `int32` | first `23` |

The fixed solver maxima are therefore contracts, not observed counts:

```text
MAX_MOLECULES             = 200
MAX_MOLECULAR_EQUATIONS   = 30
MAX_MOLECULAR_COMPONENTS  = 600
```

The pinned active equation-species vector is:

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14,
 16, 17, 19, 20, 22, 23, 24, 26, 100]
```

Index 0 is the nuclei-density scale equation. Species code `100` is the
electron equation and is active at equation index 22. The catalog contains:

- 96 polynomial molecule rows;
- 94 non-polynomial atom/ion rows;
- 18 single-component non-polynomial rows;
- 76 multi-component atomic-ion ratio rows.

The table is deliberately the synthesis catalog of 190 entries, not the
atmosphere-stage molecular catalog (`molecular_equilibrium.py:1-6`). The
textbook must not substitute one for the other.

### 2.2 Other active data

Atomic partition/Saha work is loaded from:

- `source_data_files/synthesis_tables/partition_saha_inputs.npz`
- SHA-256 `0e235e7f1edecf39630690f4c68f4fc952f55785a08174562bc9575100fc4e27`
- bytes `719700`

The fields used here include the `(6, 374)` packed partition table, `(999,)`
ionization-potential array, `(7, 56, 10, 9)` iron-group grid, `(605, 80)`
ground-partition table, stage counts for 99 elements, and the special-level
tables. Their detailed partition policy belongs to Chapter 3; Chapter 4
reuses the definitions and table conventions but recomputes the numerical
atomic state at its current electron-density seed.

Mass density uses:

- `source_data_files/synthesis_tables/atomic_masses.npz`
- SHA-256 `d4739fef7e03964aea5a7b2604f9585fd9095c26c58f5b7d5d040aaafeb5d117`
- field `atomic_mass_amu`, shape `(99,)`, `float64`
- reader: `equation_of_state.py:1438-1457`

`source_data_files/synthesis_tables/ionization_potential_lookup.npz`
(SHA-256 `54e16c51d572414ea149bc241beca774b9c3a418b339d1fbedf94da90369d135`)
is not separately opened by this call path; the active `EOSTables` bundle
already contains its `(999,)` ionization array.

## 3. Unknowns and equations

Let:

- \(D\) be the number of depth layers;
- \(M=190\) the active molecule-table rows;
- \(E=23\) the active equation rows;
- \(x_i\) be the Newton unknown at one depth, in `cm^-3`;
- \(x_0\) be the nuclei-density scale returned under the source name
  `heavy_nucleus_density`;
- \(x_e\) be the unknown at the electron equation index;
- \(A_i\) be the elemental number abundance for equation species \(i\);
- \(m_{ji}\) be the multiplicity of equation species \(i\) in table row \(j\);
- \(q_j\) be the number of inverse-electron sentinels in row \(j\);
- \(K_j(T)\) be the cgs formation constant represented by the catalog.

### 3.1 Density-space Newton, log-space product

The unknown passed to `jacrev`, `torch.linalg.solve`, and the Newton update is
the density vector itself (`molecular_equilibrium.py:548-613`).

Only the molecular mass-action product is evaluated in logarithms:

\[
\ln n_j
=
\ln K_j
+\sum_i m_{ji}\ln x_i
-q_j\ln x_e ,
\]

\[
n_j = \exp(\ln n_j).
\]

This is the exact boundary at `molecular_equilibrium.py:395-406`. `_safe_log`
first clamps each density to `torch.finfo(dtype).tiny`, the smallest normal
positive value for the active tensor dtype (`lines 356-365`). The code takes
the logarithm of the numerical cgs value; it does not nondimensionalize
densities before the logarithm.

Only rows with more than one encoded component enter the nonlinear residual.
The final post-loop evaluation uses the full component matrices, so
single atoms and ions also appear in `molecular_populations`
(`molecular_equilibrium.py:436-457`).

### 3.2 Residual rows

The code first forms

\[
R_i = x_i - A_i x_0 .
\]

It then replaces row 0 with total-particle conservation:

\[
R_0
=
\sum_{i=1}^{E-1} x_i
+\sum_{j\in\mathrm{active}} n_j
-\frac{P_{\rm gas}}{k_{\rm B}T}.
\]

For each ordinary element row, molecular stoichiometry is added:

\[
R_i
=
x_i-A_i x_0+\sum_j m_{ji}n_j .
\]

The electron row is first replaced by \(-x_e\). The implementation then adds
three catalog-encoding terms:

\[
R_e =
-x_e
+ \sum_j m_{je} n_j
+ \sum_j q_j n_j
- 2\sum_j b_j n_j ,
\]

where \(b_j\) is the `negative_ion_flag`. The last two terms encode
inverse-electron positive ions and the sign correction for negative ions.
These operations are source-exact at `molecular_equilibrium.py:378-419`.

The abundance vector used by the residual has length \(E\). Only indices
1 through \(E-1\) are populated. Catalog elements receive
`max(elemental_abundance[Z-1], 1e-20)`; index 0 stays zero
(`molecular_equilibrium.py:527-533`).

### 3.3 Formation constants

For a polynomial row with catalog coefficients \(a_0,\ldots,a_6\), the source
computes:

\[
\begin{aligned}
\ln K_j(T) ={}&
\frac{a_0}{k_{\rm B}T\,[\mathrm{eV}]}
-a_1+a_2T-a_3T^2+a_4T^3-a_5T^4+a_6T^5 \\
&-\frac{3}{2}\left(N_{\rm comp}-2q_{\rm ion}-1\right)\ln T .
\end{aligned}
\]

Here `q_ion` is obtained by rounding 100 times the fractional part of the
molecule code. The implementation is at
`molecular_equilibrium.py:181-281`.

All polynomial constants are set to zero for \(T>10000\) K. During the later
log conversion, a zero/nonpositive constant becomes the finite sentinel
`-700.0`, not `-inf` (`molecular_equilibrium.py:498-522`). Therefore the
high-temperature branch is numerically tiny, not algebraically absent.

Code `101.0` receives the special 4.478 eV polynomial at
`molecular_equilibrium.py:204-237`. Downstream code identifies the same
catalog code as molecular hydrogen (`pipeline.py:550-561` and
`payne_zero_atmosphere/synthesis_bridge.py:342-389`). The local variable
`is_hminus` at line 205 is a misleading source name; Chapter 4 must call this
species H2 and should flag the implementation name as a naming wart rather
than teaching it as H-minus.

For non-polynomial rows:

- one-component bare atom/ion entries receive \(K=1\);
- multi-component atomic-ion entries receive the supplied
  `ion_formation_constants`.

The overwrite is at `molecular_equilibrium.py:503-515`.

### 3.4 Atomic-ion formation constants

`molecular_equilibrium_metadata` identifies ion rows by:

1. zero leading polynomial coefficient;
2. component count greater than one;
3. atomic number between 1 and 99;
4. ion stage equal to `component_count - 1`.

See `molecular_equilibrium.py:1134-1175`.

For each selected row, the formation constant is:

\[
K_{\mathrm{ion},j}
=
\frac{f_{Z,q}}{f_{Z,0}}\,n_{e,\mathrm{seed}}^q ,
\]

where \(f\) is the actual atomic ion-stage fraction, recovered as
`ion_stage_fractions_over_partition * partition_functions`. The code requires
the neutral fraction to be positive; other entries remain zero. Electron
density is floored at `1e-300` before exponentiation
(`molecular_equilibrium.py:1179-1206`).

The factor \(n_e^q\) cancels the \(n_e^{-q}\) encoded in the molecule product.
For one atomic component and \(q\) inverse-electron sentinels, the result has
the intended units of number density.

## 4. The molecular electron-density seed

`molecular_seed_electron_density` does not simply copy the current electron
density (`equation_of_state.py:1340-1363`).

For the first depth:

\[
n_{e,\mathrm{seed},0}
=
\frac{P_0}{k_{\rm B,ref}T_0}\frac{1}{20}.
\]

For every later depth:

\[
n_{e,\mathrm{seed},d}
=
n_{e,d-1}\frac{P_d}{P_{d-1}} .
\]

Every value is floored at `FLOAT64_POSITIVE_FLOOR = 1e-300`.

`molecular_ion_formation_constants_from_seed` then:

1. builds an atomic derived state at that seed;
2. supplies unit abundances to the atomic Saha solver;
3. converts \(f/U\) to \(f\) by multiplying the partition functions;
4. transfers the result to host `float64`;
5. constructs the ion formation constants above.

See `equation_of_state.py:1366-1396`.

This makes input depth order part of the numerical contract. The public
builder documents outer-to-inner ordering and increasing column mass
(`payne_zero_synthesis/api.py:359-374`), but the molecular seed and continuation
functions themselves do not validate pressure monotonicity or depth order.

## 5. Exact Newton and continuation algorithm

### 5.1 Initialization

The default `chain_length=None` becomes one chain of length \(D\). A new chain
starts at depth 0, or every `chain_length` layers when an explicit value is
provided (`molecular_equilibrium.py:559-565`).

At a chain start:

```text
initial_total_density = (P / kT) / 2
if T < 4000 K:
    initial_total_density = P / kT
base_density = initial_total_density / 10
x[0] = initial_total_density
x[1:] = base_density * equation_abundance[1:]
x[electron_index] = base_density
```

At a continuation depth:

\[
\mathbf{x}^{(0)}_d
=
\mathbf{x}^{(*)}_{d-1}\frac{P_d}{P_{d-1}}.
\]

This is sequential, pressure-scaled continuation
(`molecular_equilibrium.py:565-583`).

### 5.2 Per-depth, per-iteration work

For every depth and Newton iteration:

1. evaluate the residual for that single depth;
2. call `jacrev(resid_one, argnums=0)` for that density row;
3. solve the column-scaled linear system;
4. test convergence **before** damping;
5. damp sign reversals;
6. impose the multiplicative positivity floor;
7. continue or accept the updated density.

The column-scaled solve is:

\[
s_i=\max(|x_i|,\mathrm{tiny}),\qquad
(J\,\mathrm{diag}\,\mathbf{s})\,\delta_{\rm frac}=R,\qquad
\delta=\mathbf{s}\odot\delta_{\rm frac}.
\]

This is conditioning by fractional-update columns; it does not change the
Newton variables to logarithms (`molecular_equilibrium.py:422-433`).

Convergence uses:

\[
\max_i
\frac{|\delta_i|}{\max(|x_i|,\mathrm{tiny})}
\le \mathrm{tol}.
\]

The production defaults are `max_iter=200`, `tol=1e-3` at the direct molecular
API. EOS callers pass `tol=1e-4` or the enclosing EOS tolerance.

### 5.3 Damping and floors

The exact order and constants are:

- convergence is assessed on the undamped `delta`;
- if the current `delta` reverses sign relative to the previous update for an
  unknown, multiply that component by `0.69`;
- form `candidate_density = current_density - delta`;
- if the candidate is less than `current_density / 100`, replace it with
  `current_density / 100`;
- save the damped `delta` as `previous_update`.

See `molecular_equilibrium.py:587-617`.

Other numerical guards:

- density logs use `torch.finfo(dtype).tiny`;
- abundance entries use a `1e-20` floor;
- formation logs use a `1e-300` host floor and a `-700` inactive sentinel;
- nonfinite molecular terms in the residual are replaced with zero
  (`molecular_equilibrium.py:405-406`);
- the final `_molecular_densities` call has no equivalent finite replacement.

If the molecular solve reaches `max_iter`, it records `max_iter` and returns
the last density. It does **not** raise. Iteration counts are exposed only when
`return_diagnostics=True` (`molecular_equilibrium.py:615-669`).

### 5.4 Where `vmap` is, and is not

The Newton loop is ordinary Python:

```text
for depth_index in range(n_depths):
    for iteration_index in range(max_iter):
        residual = ...
        jacobian = jacrev(...)(...)
        ...
```

After every depth loop has finished, the returned rows are stacked. Only then does
the code create:

```python
batched_moldens = vmap(_molecular_densities)
```

to evaluate final table populations over the returned depth rows
(`molecular_equilibrium.py:621-651`).

## 6. Full and fixed EOS routes

### 6.1 Full route: `solve_population_state`

Call graph:

```text
solve_population_state(..., molecules=True)
└── solve_electron_density(..., molecules=True)
    ├── atom-only charge fixed point
    ├── molecular_seed_electron_density
    ├── atomic Saha state at the molecular seed
    ├── ion_formation_constants_from_saha
    ├── solve_molecular_equilibrium          # first molecular solve
    └── atomic EOS at returned molecular ne and nuclei density
└── _assemble_population_state(..., molecules=True)
    └── _molecule_backed_population_state
        └── solve_molecular_equilibrium      # second molecular solve
```

Source: `equation_of_state.py:1467-1653,2045-2080,2150-2265`.

The preliminary atom-only electron fixed point uses number fractions and:

\[
n_{\rm nuclei}
=
\max\left(\frac{P}{k_{\rm B,ref}T}-n_e,10^{-300}\right),
\]

\[
n_{e,\rm charge}
=
\sum_{Z,q} q\,f_{Z,q}\,n_{\rm nuclei}A_Z .
\]

Its exact damping is:

```text
new_ne = max(charge_ne, current_ne / 2)
new_ne = 0.5 * (new_ne + current_ne)
```

and convergence is the relative change against `new_ne`. Unlike the
molecular Newton solve, this loop raises after `max_iter`
(`equation_of_state.py:1557-1592`).

With `molecules=True`, the first molecular solve replaces both electron density
and nuclei density in the returned `ElectronDensityResult`, recomputes the
atomic EOS at those values, and computes mass density from the molecular
nuclei scale (`equation_of_state.py:1594-1653`).

Assembly invokes the molecular solve a second time. The final public
`PopulationState` uses the second solve’s nuclei density and molecular arrays.
Its mass density is rescaled by the ratio of the second solve’s nuclei density
to the first solve’s nuclei density (`equation_of_state.py:2219-2228`).

### 6.2 Fixed route: `solve_population_state_at_electron_density`

This route:

1. keeps the supplied electron-density vector;
2. builds an atomic EOS at that vector;
3. starts with
   \(n_{\rm nuclei}=\max(P/k_{\rm B,ref}T-n_e,10^{-300})\);
4. assembles the population state;
5. when molecules are enabled, runs the molecular solver once.

Source: `equation_of_state.py:2083-2147`.

The molecular solver’s returned electron vector is deliberately assigned to
`_electron` and discarded inside `_molecule_backed_population_state`
(`equation_of_state.py:1877-1892`). Therefore:

- `PopulationState.electron_density` is exactly the caller-supplied fixed
  vector;
- `PopulationState.molecular_equation_densities[:, electron_index]` is the
  electron unknown converged by the molecular residual;
- the source has no equality assertion between those two vectors.

The molecular solver’s nuclei density is used publicly. If `mass_density` was
provided, it is preserved. Otherwise the atomic-composition density is scaled
to the molecular nuclei density (`equation_of_state.py:2124-2146,2219-2228`).

This is the atmosphere-to-synthesis bridge route. It must be described as a
population fill **at a retained atmosphere electron density**, not as a new
fully coupled electron-density solve.

### 6.3 Which route the public atmosphere builder uses

`payne_zero_synthesis.synthesis.build_structured_atmosphere_from_columns`:

1. resolves device/dtype;
2. loads `EOSTables`;
3. enables all 54 supported molecular line species when
   `molecular_lines=True`;
4. passes the atmosphere electron density as `electron_density_seed`;
5. calls the pipeline bridge.

See `payne_zero_synthesis/synthesis.py:39-87`.

The pipeline bridge then calls
`solve_population_state_at_electron_density`, selecting
`electron_density_seed` when it is supplied (`pipeline.py:345-371`). Thus the
public structured-atmosphere builder uses the fixed route, not the full route.

## 7. Shapes, axes, units, and padding

### 7.1 Molecular solver

| Quantity | Active shape | Stored/returned shape | Axes | Unit |
|---|---:|---:|---|---|
| input `temperature` | `(D,)` | `(D,)` | depth | K |
| input `gas_pressure` | `(D,)` | `(D,)` | depth | dyne cm^-2 |
| input/reference `electron_density` | `(D,)` | `(D,)` | depth | cm^-3 |
| `elemental_abundances` | `(99,)` | `(99,)` | `Z-1` | number fraction |
| `ion_formation_constants` | `(D,190)` | `(D,190)` or wider input sliced to 190 | depth, table row | cgs mass-action units |
| `component_multiplicity` | `(190,23)` | same | table row, equation row | dimensionless |
| `natural_log_formation_constants` | `(D,190)` | same | depth, table row | log numerical cgs value |
| returned density vector | `(D,23)` | internal | depth, equation row | cm^-3 |
| `equation_densities` | `(D,23)` | `(D,30)` | depth, equation row | cm^-3 |
| active molecular populations | `(D,190)` | `(D,200)` | depth, table row | cm^-3 |
| returned nuclei scale | `(D,)` | `(D,)` | depth | cm^-3 |
| returned electron density | `(D,)` | `(D,)` | depth | cm^-3 |

The last seven equation columns and last ten molecule columns are exact zero
padding when returned by the solver (`molecular_equilibrium.py:644-651`).

### 7.2 Atomic EOS and public population state

`EOSResult` uses `(D,99,6)` arrays with axes
`(depth, atomic_number_minus_one, ion_stage_minus_one)`:

- `partition_functions`: dimensionless;
- `ion_stage_fractions_over_partition`: dimensionless \(f/U\);
- `partition_normalized_populations`: \(n/U\), `cm^-3 per partition function`.

Source: `equation_of_state.py:1039-1168`.

`PopulationState` exposes:

| Field | Shape | Unit |
|---|---:|---|
| `electron_density` | `(D,)` | cm^-3 |
| `total_nuclei_number_density` | `(D,)` | cm^-3 |
| `mass_density` | `(D,)` | g cm^-3 |
| `partition_normalized_populations` | `(D,6,139)` | cm^-3 per partition function |
| `ion_stage_populations` | `(D,6,139)` | cm^-3 |
| H I, H II, He I, He II, Mg I/U, Al I/U, Si I/U, Fe I/U | `(D,)` each | cm^-3, with `/U` for named partition-normalized fields |
| H and C partition-normalized pairs | `(D,2)` each | cm^-3 per partition function |
| `molecular_populations` | `(D,200)` or `None` | cm^-3 |
| `molecular_equation_densities` | `(D,30)` or `None` | cm^-3 |
| `eos` | three `(D,99,6)` tensors | as above |

The public population cubes use axes
`(depth, ion_stage_minus_one, species_column)`, unlike `EOSResult`, whose
element axis precedes the stage axis. Atomic elements occupy species columns
0 through 98. The 40 extra columns are reserved for synthesis species mapping.

The molecule-backed path first assembles two `(D,1006)` packed schedules:

- MODE 12: actual ion-stage populations;
- MODE 11: partition-normalized per-ion populations.

It maps only stages 1 through 6 into the public cubes
(`equation_of_state.py:1928-2042`). Packed higher stages and the public
six-stage cube are distinct contracts.

## 8. Molecule-backed population assembly

`_molecule_backed_population_state` performs these exact operations:

1. collapse a two-dimensional abundance matrix to its first depth row;
2. load molecular metadata;
3. compute seed-based atomic-ion formation constants;
4. solve molecular equilibrium;
5. copy device tensors to host `float64`;
6. compute molecular and atomic-ion line populations;
7. rescale the atomic EOS fallback populations to the molecular nuclei-density
   scale;
8. fill the MODE 12 and MODE 11 packed schedules;
9. map the packed schedules into the public `(D,6,139)` cubes.

Source: `equation_of_state.py:1850-2018`.

For a molecule/ion family represented by the molecular table,
`_molecular_population_block` walks table codes from the highest requested
stage down and uses:

- `molecular_line_populations` for MODE 1 or 11;
- actual `molecular_populations` for MODE 12.

If the atomic-number family is absent from the molecule table, it falls back
to the atomic EOS state. If the family exists but a particular code is absent,
the output stays zero rather than falling back
(`equation_of_state.py:1780-1847`).

Fallback atomic populations are multiplied by:

\[
\frac{n_{\rm nuclei,molecular}}
     {\max(n_{\rm nuclei,atomic},10^{-300})}.
\]

This makes molecule-free elements share the molecular density scale.

Dedicated packed-slot exports are exact:

```text
H I, H II                         stage slots 0, 1
He I, He II                       stage slots 2, 3
H partition-normalized pair       per-ion slots 0:2
C partition-normalized pair       per-ion slots 20:22
Mg I / U                          per-ion slot 77
Al I / U                          per-ion slot 90
Si I / U                          per-ion slot 104
Fe I / U                          per-ion slot 350
```

See `equation_of_state.py:2232-2265`.

## 9. Molecular line-population boundary

The population needed by line opacity is \(N_{\rm mol}/U_{\rm mol}\), not the
raw molecular number density. The conversion is host NumPy `float64`.

For each ordinary element equation, the transformed density is:

\[
\tilde n_Z
=
\frac{n_Z}
{U_{Z,0}\,(1.8786\times10^{20})\,(A_ZT)^{3/2}}.
\]

For the electron equation:

\[
\tilde n_e
=
\frac{n_e}
{2(2.4148\times10^{15})T\sqrt{T}}.
\]

For a polynomial molecule:

\[
\frac{N_{\rm mol}}{U_{\rm mol}}
=
\exp\!\left[\frac{a_0}{T/11604.5}\right]
\left(\prod \tilde n_i\right)
(1.8786\times10^{20})(A_{\rm mol}T)^{3/2},
\]

with division by the transformed electron density for inverse-electron
sentinels. For a non-polynomial atomic-ion row, the all-species function uses
`molecular_population / partition_function_for_that_ion_stage`.

Source: `molecular_equilibrium.py:779-1008`.

The 54 supported molecular line-list species IDs are mapped to one or more
equilibrium molecule codes at `molecular_equilibrium.py:1011-1068`.
`molecular_line_populations_by_species_code` sums the mapped molecule-code
columns (`lines 1071-1114`).

## 10. Pipeline and bridge mappings

### 10.1 Synthesis-side builder

The synthesis pipeline first obtains a `PopulationState` through the fixed
route. If molecular species are requested:

1. it takes the atomic partition cube from `population_state.eos`;
2. recomputes partition functions for all molecular-equilibrium elements with
   `apply_ground_partition=False`;
3. reuses `molecular_equation_densities` and `molecular_populations` when the
   population state already contains them;
4. otherwise runs the molecular solver;
5. replaces the simple H2 estimate with molecule code `101.0` when present;
6. computes molecular line populations for the requested species IDs;
7. writes each molecular line population to:

```text
partition_normalized_populations[
    :, 5, species_code // 6 - 1
]
```

Source: `pipeline.py:438-582`.

The stage-index-5 molecular slot is a synthetic opacity lookup slot. It must
not be described as the sixth physical ion stage of the molecule.

The final structured mapping contains the population cubes, named continuum
populations, H2, Doppler widths, abundance vector, thermodynamic columns, and
continuum-edge arrays (`pipeline.py:584-635`). Its schema is version 4.

The no-ground-floor recomputation at `pipeline.py:469-487` is an important
boundary: molecular band line populations do not simply reuse the grounded
atomic partition values used elsewhere in the EOS.

### 10.2 Atmosphere-side packed bridge

When a converged atmosphere runtime state is available, the atmosphere bridge
does not call the synthesis molecular solver. It:

1. maps packed atmosphere arrays to `(D,6,139)` atomic cubes;
2. receives optional atmosphere `molecular_state` arrays;
3. fills synthetic molecular species slots from
   `partition_normalized_molecular_populations`;
4. uses actual molecule code `101.0` for H2 when available, otherwise a
   temperature-gated analytic fallback;
5. writes the same schema-v4 structured mapping.

Source: `payne_zero_atmosphere/synthesis_bridge.py:226-282,285-389,392-612`.

The atmosphere bridge requires depth-independent abundances for the structured
schema and rejects layer-dependent values exactly
(`synthesis_bridge.py:210-223`).

These are two distinct legitimate bridges:

- a live converged atmosphere can export its already-computed molecular state;
- the public synthesis column builder recomputes the synthesis molecular
  network at the retained input electron density.

Chapter 4 should not blur them into one algorithm.

## 11. Host, device, and dtype boundaries

### 11.1 Runtime policy

The public synthesis runtime prefers CUDA, then MPS, then CPU. Omitted dtype
resolves to:

- `float32` on MPS;
- `float64` on CUDA or CPU.

An explicit MPS `float64` request is rejected by `resolve_runtime`
(`device.py:13-56`).

The direct molecular solver has a slightly different local behavior: if it
receives `dtype=torch.float64` on MPS, it silently substitutes `float32`
(`molecular_equilibrium.py:475-480`).

### 11.2 Host work

The following are host NumPy `float64`:

- input canonicalization;
- molecular table loading and metadata;
- polynomial formation constants;
- molecular electron-density seed;
- atomic-ion formation constants after the atomic tensors are copied back;
- molecular line-population conversion;
- packed population assembly;
- public `PopulationState` arrays;
- structured-atmosphere packing.

The atomic EOS also retains host `float64` copies for discrete partition-table
brackets and regime gates (`equation_of_state.py:119-211,1078-1117`).

### 11.3 Device work

The following live on the selected torch device and work dtype:

- `MolecularStructure` multiplicities, powers, flags, and masks;
- log formation constants;
- abundance vector and total particle density;
- the per-depth Newton density, residual, Jacobian, and linear solve;
- the final `vmap` molecular evaluation;
- active and padded molecular output tensors;
- atomic EOS tensors and partition tables.

Caller dtype choices are not identical:

- the first molecular solve in the full EOS route requests synthesis
  `REFERENCE_DTYPE` on non-MPS, otherwise `tables.dtype`
  (`equation_of_state.py:1612-1622`);
- `_molecule_backed_population_state` uses `tables.dtype` directly
  (`equation_of_state.py:1882-1891`);
- the pipeline fallback re-solve requests `REFERENCE_DTYPE` off MPS
  (`pipeline.py:517-534`).

This means the full route can run its first and second molecular solves at
different dtypes if an off-MPS `EOSTables` instance was deliberately built in
`float32`. A parity fixture must record the dtype of both calls.

## 12. Molecular specific internal energy

There is no molecular-specific-internal-energy field or computation in the
synthesis molecular-equilibrium, `PopulationState`, synthesis pipeline, or
schema-v4 structured atmosphere.

Molecular specific internal energy is an atmosphere-solver responsibility,
implemented through the atmosphere molecular state and
`payne_zero_atmosphere/runner.py`. It must not be invented as an output of the
synthesis chemistry route. Chapter 4 may state this boundary and defer the
atmosphere thermodynamic derivative to the chapter that owns atmosphere
closure.

## 13. Golden and tolerance plan

No Chapter 4 implementation should be accepted from conservation plots alone.
The oracle must be generated by the pinned commit and record both algorithmic
internals and public outputs.

### 13.1 Fixture design

Use one deterministic, outer-to-inner, pressure-increasing atmosphere with:

- at least one layer below 4000 K, to exercise the alternate chain seed;
- layers on both sides of 10000 K, to exercise the polynomial gate;
- a realistic depth range in pressure and electron density;
- one depth-independent `(99,)` abundance vector;
- all 54 supported molecular line species for the public bridge;
- CPU `float64`, CPU `float32`, and available accelerator runs;
- a second short fixture with `chain_length=2` to test explicit restart
  boundaries without replacing the production default.

The fixture archive must carry:

- pinned Payne Zero commit;
- all active data-file SHA-256 values;
- fixture SHA-256;
- device and dtype;
- torch, NumPy, and Python versions;
- exact requested `tol`, `max_iter`, and `chain_length`.

### 13.2 Golden layers

1. **Catalog golden**
   - counts, active shapes, equation species codes;
   - component starts/indices;
   - 76 ion-formation row tuples;
   - exact zero padding.

2. **Formation golden**
   - host `formation_constants`;
   - `natural_log_formation_constants`;
   - seed electron density;
   - ion-stage fractions used to form ion constants;
   - `ion_formation_constants`.

3. **Molecular Newton golden**
   - returned active density rows `(D,23)`;
   - iteration count at every depth;
   - padded `(D,30)` equation densities;
   - padded `(D,200)` molecular populations;
   - returned electron and nuclei vectors;
   - final normalized residual by row family.

4. **Full-route golden**
   - first molecular solve outputs;
   - second molecular solve outputs;
   - all `ElectronDensityResult` fields;
   - every `PopulationState` field;
   - packed MODE 11 and MODE 12 schedules.

5. **Fixed-route golden**
   - exact retained public electron density;
   - molecular equation electron density as a separately named field;
   - supplied versus derived mass-density branches;
   - every public state field.

6. **Bridge golden**
   - all schema-v4 arrays;
   - H2 source selection;
   - all 54 requested synthetic molecular slots;
   - a ground-floor-on versus ground-floor-off line-population checkpoint;
   - exact untouched zeros in unused cube cells.

### 13.3 Acceptance metrics

Use field-complete comparison: numeric key sets, shapes, axes, and dtypes must
match before numeric metrics are evaluated. Missing fields, unexpected fields,
NaNs, and infinities are hard failures.

For CPU `float64`:

- require exact equality for metadata, indices, masks, padding, branch
  decisions, iteration counts, and all structural zeros;
- require bitwise identity for source-staged functions whose operation order is
  unchanged;
- for a deliberately decomposed pedagogical implementation, freeze a strict
  measured tolerance only after comparing against the bitwise source-staged
  oracle; do not choose a permissive tolerance in advance.

For `float32`/accelerator:

- compare absolute, scale-relative, and resolved-relative errors separately;
- compare `log10` values for positive formation constants and populations that
  span many decades;
- require zero leakage to remain exactly zero in padded and inactive cells;
- freeze thresholds from a recorded baseline for each backend, with explicit
  margin, rather than copying the Chapter 3 atomic tolerance.

Independent physical residual gates must include:

- total-particle closure;
- every active elemental conservation row;
- charge closure;
- nonnegative finite densities;
- mass-density consistency;
- public fixed-electron identity;
- packed-to-public cube identity;
- synthetic species slot identity.

The molecular solver’s update tolerance is not itself a residual tolerance.
Both must be recorded.

## 14. Unresolved priority points

### P0

No unresolved source-interpretation P0 remains after this audit.

The previously blocking global-plan statements are resolved as follows:

- Newton unknowns are densities, not logarithms;
- molecular products alone use log evaluation;
- both full and fixed synthesis routes solve depth layers sequentially;
- `jacrev` is per depth and per iteration;
- `vmap` is only the post-depth-loop final-population evaluation.

These corrections remain P0 gates for any Chapter 4 outline, notebook, or
implementation.

### P1

1. **Fixed-electron route has two electron vectors.** The public electron
   density is retained from the input, while the molecular equation density has
   its own solved electron component, and the source does not assert equality.
   The golden must expose both. Chapter prose must not call this route fully
   electron-self-consistent.

2. **The full route performs two molecular solves.** The first is inside
   `solve_electron_density`; the second is during population assembly. There is
   no explicit equality check between their states. The golden must retain both
   rather than observing only the final `PopulationState`.

3. **Molecular nonconvergence is not an exception.** `max_iter` returns the last
   state. The textbook runtime must surface iteration counts and residuals and
   must fail its own acceptance gate when a fixture reaches `max_iter`.

4. **Nonfinite residual terms are zeroed.** Overflow in a molecular term can be
   hidden by `torch.where(isfinite, term, 0)`, while the final population
   evaluation has no matching guard. Finiteness checks are mandatory around
   both boundaries.

5. **Depth-dependent abundance input is not supported consistently.**
   `_molecule_backed_population_state` silently uses only the first abundance
   row, while the atmosphere bridge explicitly rejects layer-dependent
   abundances and the public synthesis builder documents `(99,)`. Chapter 4
   should accept and teach the `(99,)` contract only.

6. **Code `101.0` is H2 despite the local `is_hminus` variable name.** The
   physical interpretation should follow the downstream H2 mapping and the
   4.478 eV formula; the misleading implementation name should be noted once,
   not propagated.

7. **The 10000 K polynomial cutoff is represented by `-700`, not an exact
   zero.** Claims that molecular terms are algebraically removed above the gate
   would be false. A boundary golden is required.

8. **Continuation assumes meaningful depth order and pressure ratios.** The
   low-level solver does not validate them. The chapter-facing wrapper should
   validate positive finite temperature and pressure, outer-to-inner ordering,
   and the intended continuation order before entering Newton.

9. **The line-population bridge changes the partition policy.** Synthetic
   molecular band slots use neutral partitions recomputed without the ground
   floor. A golden must prevent an implementation from reusing the ordinary
   grounded partition cube by convenience.

10. **Dtype can differ between the two full-route molecular solves.** This can
    occur for off-MPS `float32` tables because the first solve explicitly asks
    for reference `float64`, while assembly uses `tables.dtype`. The chapter
    implementation must make this boundary visible and test it.

## 15. Chapter 4 implementation constraints derived from this audit

These are source gates, not a proposed prose outline:

- introduce the 23 density unknowns before showing any log expression;
- label log evaluation as an overflow-control boundary around mass-action
  products;
- show a single-depth residual before the continuation loop;
- implement the exact column-scaled density Newton step;
- preserve convergence-before-damping order and the literals `0.69` and
  `/100`;
- show the sequential depth loop explicitly;
- introduce `jacrev` only after the residual is readable;
- introduce `vmap` only after every depth loop terminates, for final molecular populations;
- distinguish the full and fixed routes in names, diagrams, outputs, and
  goldens;
- preserve all active/padded dimensions and public field names;
- teach the synthesis catalog of 190 molecules as distinct from the atmosphere
  catalog;
- retain the exact synthesis constants and do not merge them with the
  atmosphere constants;
- treat population packing and synthetic stage-5 molecular slots as a data
  contract, not as incidental indexing;
- state explicitly that synthesis chemistry does not compute molecular
  specific internal energy.
