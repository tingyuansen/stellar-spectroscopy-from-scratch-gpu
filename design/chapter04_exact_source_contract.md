# Chapter 4 Exact Source, Data, and Teaching Contract

Status: frozen pre-draft contract for **Molecules and Coupled Equilibrium**  
Pinned implementation: Payne Zero commit
`9c44001feae40b85146630499e6f8a5fed42e5af`  
Oracle policy: read-only; no reader computation may import a golden as input

This document is the central authority for Chapter 4. It reconciles:

- `design/chapter04_atmosphere_source_audit.md`;
- `design/chapter04_synthesis_source_audit.md`;
- `design/chapter04_ownership_data_audit.md`;
- `design/global_chapter_contracts.md`;
- Chapters 2 and 3's accepted data, layout, and atomic-state contracts.

The audit files retain source-level detail. This contract owns the chapter's
reader-facing claims, exact implementation boundaries, staging order, golden
design, and acceptance gates.

## 1. The causal question

Chapter 3 could close each element's ion ladder separately because no particle
contained nuclei from two elements. CO breaks that independence: one CO
molecule spends one carbon nucleus and one oxygen nucleus at the same time.

Chapter 4 answers:

> Once several elements may bind into the same molecule, how can mass action,
> elemental conservation, particle conservation, and charge determine one
> positive thermochemical state at every ordered depth?

The chapter consumes, without re-deriving:

- Chapter 3's level, partition-function, Saha, and atomic-population
  definitions;
- the distinction between actual and partition-normalized populations;
- the atmosphere `(depth,1006)` and public synthesis `(depth,6,139)` layouts;
- the declared elemental abundance vector and data-manifest discipline.

The chapter recomputes the numerical atomic, electron, nuclei, mass, and
molecular state. It does not treat Chapter 3's atom-only numerical state as the
answer after molecules are enabled.

Chapter 5 receives checked populations and densities. Chapter 4 must not
preview continuum cross-sections or opacity formulae.

## 2. Non-negotiable source corrections

The following statements are P0 gates:

1. Both production molecular solvers use ordinary **number densities** as
   Newton unknowns. Neither solves for log densities.
2. Only the synthesis mass-action products are evaluated in logarithms.
3. Both solvers advance depths sequentially. Ordinary atmosphere population
   mode and every synthesis route seed a new depth by pressure-scaling the
   preceding returned solution. Atmosphere specific-internal-energy mode first
   applies that scaling but replaces it with a nonzero saved physical-density
   row when available. A returned solution may be the last iterate after
   exhaustion; chapter acceptance rejects that case.
4. The atmosphere path has no molecular `prange` or GPU solve.
5. Synthesis calls `jacrev` for one depth on every Newton iteration.
6. Synthesis uses `vmap` only after every ordered depth loop has terminated
   (by convergence or exhaustion) to evaluate final molecular populations.
7. The molecule-enabled atmosphere structured handoff does not preserve its
   supplied electron density. It runs coupled equilibrium, then performs a
   fixed packed-population refill at the newly solved density.
8. The synthesis fixed route preserves its **public** input electron-density
   field, but its internal molecular equation still solves an electron
   unknown. These two arrays are not required to agree.
9. The synthesis full route performs two molecular solves. Its published
   electron density and its final molecular arrays can therefore come from
   different solves.
10. The live/debug atmosphere molecular bridge is shape-incompatible in the
    pinned wrapper: padded molecule codes `(200,)` are paired with active
    populations `(depth,170)`. A chapter adapter may slice active codes for a
    diagnostic, but must label that adapter and must not claim the pinned
    wrapper already succeeds.
11. The release product is rebuilt through the synthesis implementation and
    its 190-entry catalog; it is not a serialization of the atmosphere
    solver's 170-entry molecular state.
12. Reaching a molecular Newton iteration limit returns the last state rather
    than raising. Chapter diagnostics must surface this limitation without
    changing the exact parity result.

## 3. Exact source identities

| Source | SHA-256 | Chapter 4 responsibility |
| --- | --- | --- |
| `payne_zero_atmosphere/molecular_data.py` | `705c3072d79c8019c948ce0fa2c82052f232816d453e10a7c8e5fc5a8f5ce249` | atmosphere catalog and base-100 codec |
| `payne_zero_atmosphere/molecular_equilibrium.py` | `4c9665148a57fadf4837f193c9f4247fbab44b788e7fb76ef3cae7ebdb3c3d86` | atmosphere chemistry, continuation, populations, energy |
| `payne_zero_atmosphere/equation_of_state.py` | `719b316327fd6fb76dfa5267a2c9022d1384c269a0fa20d5b2ec733671ae3fa2` | atmosphere population dispatch and solve cache |
| `payne_zero_atmosphere/runner.py` | `05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7` | full and structured population routes |
| `payne_zero_atmosphere/population_layout.py` | `36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0` | 1006-slot atomic/molecular schedule |
| `payne_zero_atmosphere/synthesis_bridge.py` | `142a960b5e710823754b02766803b3c1dd8c48c9945fdfabe560b4ee7e1acb50` | live packed-to-public mapping and release boundary |
| `payne_zero_synthesis/molecular_equilibrium.py` | `df01757c160b2bff4390cc2148cff9d1ba6e5a2bc7cab4515b46f38e868d2714` | synthesis chemistry, `jacrev`, final `vmap`, line normalization |
| `payne_zero_synthesis/equation_of_state.py` | `6497c29abb954e0b55d918cc22fa7b660952812c548faf1d7b1053345ef13562` | molecular seeds and full/fixed public states |
| `payne_zero_synthesis/pipeline.py` | `465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22` | fixed-public-\(n_e\) structured bridge and synthetic lane |
| `payne_zero_synthesis/atmosphere_schema.json` | `2ba8d637e613be12ff43ce319a752616323f0341ea69f8e2391c3c244939777a` | schema-v4 public arrays |

## 4. Complete static data contract

Copy these complete byte-identical assets, not subsets:

| Textbook destination under `data/static/` | Bytes | SHA-256 | Active content |
| --- | ---: | --- | --- |
| `source_catalogs/lines/molecular_equilibrium_atmosphere.npz` | 19,040 | `971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644` | 170 rows, 23 equations, 481 components |
| `source_catalogs/lines/molecular_equilibrium_synthesis.npz` | 18,060 | `3e8c1ea69fe672b9886bda38922f868c6d2ac2b43c4eb0d7750620241c238d28` | 190 rows, 23 equations, 548 components |
| `atmosphere_tables/molecular_equilibrium_tables.npz` | 1,935 | `1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe` | atmosphere H2 partition table `(200,)`; atomic masses retained but not made a new authority |
| `synthesis_tables/continuum_edge_grid.npz` | 24,104 | `11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e` | six exact edge-grid arrays plus provenance bytes required by the public structured builder; opacity use remains deferred |

The chemistry also inherits two accepted Chapter 3 dependencies:

| Local asset | Local SHA-256 | Pinned source identity | Role |
| --- | --- | --- | --- |
| `data/static/synthesis_tables/partition_saha_tables.npz` | `83e7708b0ca989caca05532ea701318d7962c3a054e48b5529d69780fc6c1f70` | role-separated from `partition_saha_inputs.npz`, source SHA-256 `0e235e7f1edecf39630690f4c68f4fc952f55785a08174562bc9575100fc4e27` | invariant atomic partition/Saha inputs |
| `data/static/synthesis_tables/atomic_masses.npz` | `d4739fef7e03964aea5a7b2604f9585fd9095c26c58f5b7d5d040aaafeb5d117` | byte-identical source | synthesis mass density, bridge mean mass, and atomic Doppler support |

Molecular line-population normalization does **not** read
`atomic_masses.npz`. It uses the parity-rounded hard-coded
`_ATOMIC_MASSES_FOR_MOLECULES` array in the exact synthesis molecular module.
That `(99,)` `float64` array has raw-byte SHA-256
`b6b870f0cdb3ea49fc4977dfc1fcff0ed1c16747922940d302644aefe28b7636`.
The chapter helper and parity tests use this exact authority; substituting the
more precise NPZ masses is a parity error.

The synthesis molecular algebra uses exact
`BOLTZMANN_ERG_PER_K=1.380649e-16` and
`BOLTZMANN_EV_PER_K=8.617333262e-5`. Its inherited atomic EOS deliberately
retains the rounded reference tier, including
`REFERENCE_BOLTZMANN_ERG_PER_K=1.38054e-16`. These are not merged.

The common active equation-species codes are:

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14,
 16, 17, 19, 20, 22, 23, 24, 26, 100]
```

Index 0 is the nuclei-density scale. Species code 100 is the electron
equation. Atmosphere code 101 in a component list is the one-past-the-end
inverse-electron sentinel.

The catalogs share 170 physical records when aligned by molecule code and
component semantics. They do not share row order. The synthesis catalog has 20
additional codes:

```text
111, 10811, 10812, 10820, 60606, 60608, 60614, 60816, 61414, 61616,
70708, 70808, 80814, 80816, 1010106, 1010107, 1010606, 6060707,
101010106, 101010114
```

Production arrays must remain in production order. Scientific cross-catalog
comparisons key-align by rounded molecule-code identity; they never compare
the first 170 rows.

## 5. Molecular code and physical unknown

The atmosphere provenance codec uses eight base-100 fields. Each two-digit
block denotes an atomic number. Repeated blocks encode stoichiometric
multiplicity. A decoded zero becomes species code 100, an ordinary electron
component used for a negative ion. Positive ion charge is encoded by appending
one code-101 inverse-electron sentinel per charge.

At one depth, let:

- \(x_0\) be the total-nuclei density scale in `cm^-3`;
- \(x_i\) be a free elemental basis density in `cm^-3`;
- \(x_e\) be the electron equation density in `cm^-3`;
- \(A_i\) be an elemental number abundance;
- \(\nu_{mi}\) be the multiplicity of component \(i\) in record \(m\);
- \(p_m\) be the number of inverse-electron sentinels;
- \(\eta_m\) be the number of ordinary code-100 electron components;
- \(K_m(T)\) be the catalog formation constant in its exact cgs mass-action
  units.

The molecular population is

\[
n_m = K_m(T)
      \left(\prod_i x_i^{\nu_{mi}}\right)x_e^{\eta_m-p_m}.
\]

The production residual contains:

\[
R_0 =
-\frac{P_{\rm gas}}{k_{\rm B}T}
+\sum_{i=1}^{E-1}x_i
+\sum_m n_m ,
\]

\[
R_i = x_i-A_i x_0+\sum_m \nu_{mi}n_m ,
\]

and a charge row beginning with \(-x_e\), then adding the positive-ion and
negative-ion terms encoded by the component lists. One-component records do
not enter the nonlinear molecular sum in the residual; their free basis
density is already in the particle sum.

The atmosphere catalog has 14 negative-ion records. In each, the ordinary
electron is the last component and no inverse-electron sentinel is mixed into
the same record. Catalog acceptance pins that ordering because the atmosphere
charge correction depends on it.

This is a density-space root. Logarithms used to evaluate a product do not
change the unknown.

## 6. Formation-constant policies

### Atmosphere

- Code 101.0 is H2.
- Its special branch uses the `(200,)` tabulated H2 partition function, the
  dissociation wavenumber `36118.11 cm^-1`, an interpolation clamp at
  100–19900 K, and an activity gate through 20000 K.
- The exact 19900 K index/fraction behavior must be tested; a generic
  interpolation helper is not accepted without parity proof.
- Other nonzero-coefficient records use the six-coefficient atmosphere
  expression through 10000 K and become exactly zero above 10000 K.
- One-component zero-first-coefficient records receive one.
- Multi-component zero-first-coefficient records receive Saha-derived
  constants built from the pre-solve runtime electron and charge-square
  densities.
- Constants remain frozen during a one-depth Newton solve.
- Raw populations recompute constants once after the solved electron density
  has been installed.
- A nonfinite H2 constant is explicitly converted to zero. The ordinary
  polynomial branch calls `exp(exponent)` without an overflow or finiteness
  guard. Positive finite thermodynamic inputs do not by themselves preclude a
  pathological coefficient exponent; the chapter records this asymmetry and
  uses a controlled malformed-copy diagnostic without altering the canonical
  catalog.

### Synthesis

- Polynomial formation constants use the synthesis seven-coefficient
  expression.
- Code 101.0 is physically H2 even though one local implementation boolean is
  misleadingly named `is_hminus`.
- Polynomial constants are zeroed above 10000 K, then represented by a finite
  log sentinel `-700`, so their numerical products are tiny rather than
  algebraically absent.
- Single-component non-polynomial rows receive one.
- Multi-component atomic-ion constants are built from Chapter 3's atomic
  fractions and the seed electron density.
- Molecular products use dtype-safe log evaluation; the densities passed to
  Newton and `jacrev` remain ordinary densities.

The internal synthesis molecular seed is not the fixture's declared
`electron_density_seed`. At the first depth,

\[
n_{e,\mathrm{mol\,seed},0}
=\frac{P_0}{k_{\rm B,ref}T_0}\frac{1}{20}.
\]

At later depths it is the preceding upstream electron density multiplied by
`gas_pressure[d]/gas_pressure[d-1]`; every value is floored at `1e-300`.
Atomic-ion formation rows then use
\(K_{Z,q}=(f_{Z,q}/f_{Z,0})n_{e,\rm seed}^{q}\).

The named synthesis `molecular_hydrogen_population` also has a provisional
4.478 eV, 9000 K-gated estimate that is replaced by solved code 101 when
molecules are active. The three H2 policies must be shown once in a compact
table and never collapsed into one universal helper.

## 7. Exact atmosphere Newton algorithm

The atmosphere route is CPU NumPy `float64`.

1. Build formation constants once for the current depth.
2. Assemble the analytic residual and Jacobian in
   `@njit(cache=True,nogil=True)` kernels.
3. Solve `J delta = R` with `np.linalg.solve`.
4. Fall back to `np.linalg.lstsq(..., rcond=None)` only after
   `np.linalg.LinAlgError`.
5. Test each pre-damping relative update
   `abs(delta_i)/max(abs(x_i),1e-300)` against exactly `1e-4`.
6. Multiply a sign-flipping update by exactly `0.69`.
7. Form `updated = x_i-delta_i`.
8. If `abs(updated) >= x_i/100`, assign `abs(updated)`.
9. Otherwise assign `x_i/scale`, where the shared vector-order-dependent
   `scale` starts at 100 and becomes `sqrt(scale)` for later indices if the
   same update also changed sign.
10. Apply even the final below-tolerance update.
11. After 200 iterations, return the last vector without a convergence flag.

The atmosphere absolute-value reflection and shared mutable `scale` are exact
algorithmic facts. They must not be replaced by the synthesis floor.

The first depth starts from half the ideal particle density, or the full ideal
particle density below 4000 K, with the electron/basis seed at one tenth of
that scale. Every later depth starts from the preceding returned vector
multiplied by `gas_pressure[d]/gas_pressure[d-1]`.

Specific-internal-energy mode is a deliberate exception to the final seed at a
depth. Pressure scaling still occurs first, but any nonzero row in
`previous_molecular_equation_densities[d]` replaces the scaled equation vector
before Newton. The runtime electron seed used to build frozen Saha-derived
constants remains the pressure-scaled runtime value. After all depths, this
mode:

- does not overwrite `previous_molecular_equation_densities`;
- does not transform equation columns or fill
  `partition_normalized_molecular_populations`;
- evaluates molecular specific internal energy; and
- returns immediately.

There is no molecular `parallel=True`, `prange`, or GPU path. Numba compiles
the constant, residual/Jacobian, update, and energy kernels; LAPACK and the
ordered depth loop remain outside those kernels.

## 8. Exact synthesis Newton algorithm

The synthesis route uses the selected Torch device/dtype.

1. Build active structure and finite log formation constants.
2. Solve depths in an ordinary Python loop.
3. At each depth and iteration, evaluate one residual.
4. Call `torch.func.jacrev` on that one-depth density residual.
5. Column-scale by
   \(s_i=\max(|x_i|,\mathrm{tiny})\) and solve
   \((J\,\mathrm{diag}(s))\delta_{\rm frac}=R\).
6. Recover \(\delta=s\odot\delta_{\rm frac}\).
7. Test convergence on the undamped relative update.
8. Multiply sign-flipping components by exactly `0.69`.
9. Form `candidate=x-delta`.
10. Replace any candidate below `x/100` by `x/100`; there is no atmosphere
    absolute-value recovery and no cross-index `sqrt(scale)` state.
11. Return the last state on iteration exhaustion.
12. After all depth loops terminate, and only then, use `vmap` to evaluate the
    final molecular populations. This still occurs after an exhausted loop;
    chapter acceptance separately rejects exhaustion.

During residual construction, nonfinite molecular terms are replaced by zero.
The final post-loop molecular evaluation has no equivalent guard. The chapter
must record pre-replacement overflow diagnostics as well as final finiteness;
an output-only finite check cannot prove the residual was healthy.

The direct solver default is `tol=1e-3`; production EOS wrappers use the
molecule-backed route at `1e-4`. Every visible call names the tolerance owner.

The default chain is the complete depth column. An explicit `chain_length`
inserts cold-restart boundaries. Continuation depths use the previous detached
solution scaled by the gas-pressure ratio. Depth order is therefore part of
the numerical contract.

The chapter-facing wrapper accepts only a depth-independent `(99,)` abundance
vector, validates finite nonnegative entries with a positive sum without
renormalizing them, and validates positive finite temperature and pressure
plus the declared outer-to-inner order before entering Newton. This is a
protective teaching wrapper: the low-level continuation does not validate
those conditions, and molecule-backed synthesis silently takes the first row
of a depth-dependent abundance matrix while the atmosphere bridge rejects such
variation.

The exact synthesis network then constructs
`equation_abundance[i] = max(A_Z,1e-20)` for each active element. At every chain
start it seeds

\[
x_0=\frac{1}{2}\frac{P}{k_{\rm B}T},
\qquad
x_0=\frac{P}{k_{\rm B}T}\ \text{when }T<4000\ {\rm K},
\]

sets `base=x0/10`, assigns `x_i=base*equation_abundance[i]`, and sets the
electron equation to `base`. These exact floor and cold-chain rules are used
for the first depth and every explicit `chain_length` restart; they are not
inferred from the fixture's separate electron seed.

Device/dtype policy is route-specific:

- public runtime selection prefers CUDA, then MPS, then CPU;
- direct `solve_molecular_equilibrium` defaults to
  `DEFAULT_DTYPE=torch.float32` even on CPU and CUDA;
- public APIs with omitted dtype resolve to `float64` away from MPS, so omitted
  dtype means different things at the two interfaces;
- public MPS `float64` is rejected, while the direct molecular solver silently
  substitutes MPS `float32`;
- the full route's first molecular solve requests reference `float64` off MPS,
  while its second solve uses `tables.dtype`;
- the fixed route uses `tables.dtype`.

CPU `float64`, off-MPS `float32`, and available MPS behavior therefore need
separate route tests. A generic accelerator tolerance does not close these
seams. Every chapter and golden call to the direct solver passes dtype
explicitly; one separate interface test records the direct omitted-dtype
default.

## 9. Shapes, padding, units, and state lifecycle

### Atmosphere state

For \(D\) depths:

| Field | Shape | Meaning |
| --- | --- | --- |
| `molecular_populations` | `(D,170)` | raw `cm^-3` |
| `partition_normalized_molecular_populations` | `(D,170)` | line-population basis |
| `molecular_equation_densities` | `(D,23)` | physical densities immediately after Newton; later columns 1 onward are transformed in place |
| `previous_molecular_equation_densities` | `(D,23)` | saved physical warm-start copy |
| packed actual populations | `(D,1006)` | `cm^-3` |
| packed partition-normalized populations | `(D,1006)` | `cm^-3/U` |

Column 0 of `molecular_equation_densities` remains the physical nuclei scale.
In ordinary population modes, elemental/electron columns are transformed in
place into translational/partition-normalized basis values. The saved
`previous_...` array retains physical densities. Every table, plot, and golden
must state which lifecycle it uses.

The post-Newton lifecycle has three distinct exits:

1. Specific-internal-energy mode leaves the saved physical warm starts
   untouched, skips every normalized fill, computes energy, and returns.
2. Ordinary `population_mode in (2,12)` saves the new physical equation
   densities and returns before transforming columns or filling normalized
   molecular populations.
3. Other ordinary population modes save physical equation densities, transform
   equation columns 1 onward in place, and then fill
   `partition_normalized_molecular_populations`.

Raw `molecular_populations` are evaluated before all three exits. For a
nonzero-coefficient molecular record, the raw formation constant is gated to
zero above 10000 K (or above 20000 K for atmosphere H2), but the separate
partition-normalized line-population branch has no corresponding temperature
gate. It reconstructs its own value from the first coefficient, transformed
equation columns, the `1.8786e20` translational factor, `T/11604.5`, and
molecular mass. Raw populations, transformed equation columns, and
partition-normalized populations are therefore separate observables and must
be pinned independently on both sides of the exact gates.

The full molecule-enabled packed schedule has 230 jobs: 198 atomic plus 32
entries for 16 selected molecule codes. Molecular modes 1 and 11 both write
the same partition-normalized source; this is a verified redundant schedule
entry, not two observables.

Atmosphere lifecycle gates also prove that:

- `charge_square_density` remains the pre-solve seed during molecular Newton;
- Saha-derived constants are frozen inside the depth solve;
- population constants are recomputed after the final runtime electron density
  is installed;
- `prepare_population_state` leaves atomic and molecular populations empty
  when `pressure_iteration_enabled=False`, even though Doppler support is
  computed afterward.

### Synthesis state

| Field | Active shape | Public/stored shape |
| --- | ---: | ---: |
| returned equation densities | `(D,23)` | `(D,30)`, last 7 exactly zero |
| molecular populations | `(D,190)` | `(D,200)`, last 10 exactly zero |
| atomic EOS arrays | `(D,99,6)` | same |
| public actual/normalized cubes | — | `(D,6,139)` |

The public cube axes are `(depth, stage, species_column)`, unlike the internal
atomic EOS axes `(depth, element, stage)`.

For ordinary atomic columns, stage index 5 can mean the sixth ion stage. For
54 molecular line species, `partition_normalized_populations[:,5,column]` is
a synthetic molecular-population lane. The actual ion cube does not become a
molecular cube. One required example is:

```text
line species code 276 -> equilibrium molecule code 608 (CO)
public column = 276 // 6 - 1 = 45
```

Chapter 8 consumes this lane without re-teaching its construction.

The synthesis lane value is not raw `n_m`. Before mapping, the pipeline
recomputes the relevant atomic partitions with
`apply_ground_partition=False`, then applies the host-`float64` molecular
line-population normalization (including the `1.8786e20` translational factor,
the `T/11604.5` energy scale, the exact hard-coded parity-rounded molecular
mass array, and the transformed electron density). Acceptance includes a
ground-floor-on versus ground-floor-off value checkpoint and the hard-coded
mass-array identity, not just an index check.

## 10. Full, fixed, diagnostic, and release claims

| Route | Electron-density ownership | Molecular state source | Safe claim |
| --- | --- | --- | --- |
| Atmosphere `prepare_population_state`, molecules on, pressure iteration enabled | seed is overwritten | one ordered atmosphere solve | full molecule-enabled atmosphere population phase |
| Atmosphere `prepare_population_state`, pressure iteration disabled | seed remains in the fresh runtime state | molecular state may be allocated, but no atomic or molecular population fill runs | empty population phase followed by Doppler-support computation; not a fixed-density fill |
| Atmosphere `prepare_structured_handoff_population_state`, molecules on | input is not preserved | one ordered solve, then fixed packed refill | coupled solve followed by fixed refill |
| Synthesis `solve_population_state`, molecules on | first molecular solve supplies published \(n_e\) | second molecular solve supplies molecular arrays and molecule-backed packed state | production full route with an explicit two-solve seam |
| Synthesis `solve_population_state_at_electron_density`, molecules on | public \(n_e\) preserves input exactly | internal solve has its own electron equation; returned electron is discarded by outer state | fixed public \(n_e\), no molecular charge-closure claim |
| Public synthesis structured builder with molecular lines | upstream atmosphere \(n_e\) enters one molecular fixed route | the fixed route solves once; pipeline reuses those returned molecular arrays and maps all 54 lanes | synthesis-ready structure at upstream-owned \(n_e\), with an exact one-solve/reuse call-count claim |
| Pipeline fallback for an externally inconsistent or atom-only population state | depends on supplied state | performs a separate molecular re-solve only when molecular arrays are absent | source branch, not the standard molecular-lines public route |
| Atmosphere live/debug bridge | current wrapper fails active/padded molecule-code validation | chapter may probe a clearly labeled active-code slice | diagnostic mapping contract, not release parity |
| Atmosphere release product | final atmosphere columns are inputs | independent synthesis rebuild with 190-row catalog | release/product parity belongs to synthesis output |

This matrix appears only after the reader has executed the relevant routes. It
summarizes evidence; it is not a substitute for computation.

The live/debug bridge has an additional global H2 selection seam. If **any**
depth of catalog code 101 is positive, `_molecular_hydrogen_population`
returns the complete catalog vector unchanged, including zeros at other
depths. The analytic 9000 K-gated fallback runs only when every catalog H2
entry is nonpositive. It is not a depth-by-depth gap filler. The bridge golden
must include a mixed positive/zero catalog vector as well as the all-zero
fallback case.

The public synthesis structured builder always also loads the exact continuum
edge grid while assembling schema-v4 output. Chapter 4 does not teach opacity
from those arrays, but release/product parity cannot bypass their
byte-identified data dependency.

## 11. Molecular specific internal energy

The atmosphere route owns molecular specific internal energy. The synthesis
chemistry state and schema do not expose such a field.

An ordinary atmosphere molecular population solve assigns the translational
value \(1.5P_{\rm gas}/\rho\). The full molecular energy is evaluated in a
separate enabled mode that:

- restores saved physical equation-density warm starts;
- uses temperature samples at `T*1.001` and `T*0.999`;
- includes H2, polynomial-molecule, and atomic/ionic branches;
- divides the energy density by `max(mass_density,1e-300)`.

Entry-point defaults are intentionally asymmetric. Public
`AtmosphereConfig.molecular_convection_thermal_tracks_perturbation` defaults to
`True`, while the lower-level `compute_convection_finite_difference_samples`
and `finalize_transfer_state` parameters default to `False`.
`run_atmosphere_model` explicitly forwards the public configuration flag.
Chapter examples therefore enter through the product runner or pass the
lower-level flag explicitly; an omitted lower-level argument must not be
described as the public default. The ownership ledger keeps the orchestration
flag primarily in Chapter 12 while Chapter 4 pins its chemistry-side effect.

Chapter 4 teaches the physical meaning, saved-state lifecycle, and one checked
value. Chapter 12 owns the four convection perturbations
`T+`, `T-`, `P+`, `P-` and must not re-derive this chemistry.

## 12. Controlled fixture

Create `data/fixtures/chapter04_molecular_inputs.npz`. It contains inputs only:

- the exact Chapter 3 `(99,)` linear abundance vector;
- outer-to-inner `column_mass = geomspace(1e-6,1e2,6)`;
- `temperature = [3300,3800,4500,5500,7000,9000]` K;
- `gas_pressure = [1e2,1e3,1e4,1e5,1e6,1e7]` dyn cm\(^{-2}\);
- `microturbulence = 2e5` cm s\(^{-1}\);
- an electron seed
  `0.1*gas_pressure/(REFERENCE_BOLTZMANN_ERG_PER_K*temperature)`;
- explicit units, axis names, source identity, and fixture checksum.

Call it a controlled thermochemical track, never a converged atmosphere.

Also retain independent one-depth controls:

- temperature: `[3500,6000,9000]` K at `1e5` dyn cm\(^{-2}\);
- pressure: `[1e2,1e4,1e6]` dyn cm\(^{-2}\) at 3500 K.

These controls establish temperature and pressure directions without a
continuation history. The main track separately compares production
continuation with deliberate restarts.

Store exact branch-boundary probes as independent inputs:

- atmosphere polynomial: 10000 K and the next representable value above it;
- synthesis polynomial/log sentinel: 10000 K and the next representable value
  above it;
- named synthesis H2: 9000 K and the next representable value above it;
- atmosphere H2 partition/activity: 100, 101, 19899, 19900, 20000 K, and the
  next representable value above 20000 K.

A plotting grid does not replace these deterministic branch probes.

Named probes include H2 101, C2 606, CN 607, CO 608, N2 707, NO 708, and O2
808.

## 13. Golden products and comparison discipline

Generate comparison-only goldens in a fresh CPU `float64` subprocess with:

- the pinned commit imported first;
- one Torch, BLAS, and Numba thread;
- `PYTHONHASHSEED=0`;
- exact module/data/fixture versions;
- deterministic NPZ writing.

Create:

1. `chapter04_molecular_constants_cpu_float64.npz`;
2. `chapter04_atmosphere_molecular_state_cpu_float64.npz`;
3. `chapter04_synthesis_molecular_full_cpu_float64.npz`;
4. `chapter04_synthesis_molecular_fixed_cpu_float64.npz`;
5. `chapter04_molecular_public_mapping_cpu_float64.npz`.

Together they retain:

- both catalogs' active metadata and code-key alignment;
- H2 and polynomial branch checkpoints;
- formation constants and molecular electron seeds;
- atmosphere raw/normalized/saved equation arrays;
- all atmosphere runtime and packed population fields;
- atmosphere handoff input and output electron density;
- raw, transformed-equation, and partition-normalized atmosphere arrays on
  both sides of the 10000 K and 20000 K gates;
- both synthesis full-route molecular solves;
- synthesis published and internal electron columns;
- fixed-route supplied/published/internal electron columns;
- supplied-mass and derived-mass fixed branches;
- active and padded molecular arrays;
- every public state field;
- all 54 molecular species/molecule/public-column triples;
- live-bridge H2 selection for a mixed positive/zero vector and for an all-zero
  vector;
- untouched structural zeros and exact padding;
- iteration counts and normalized residuals.

The public structured-builder trace additionally proves that the
molecular-lines route calls the fixed molecular solve once, reuses its
returned molecular arrays, does not enter the pipeline fallback re-solve, and
loads the exact continuum edge grid.

Because the public full `PopulationState` does not expose every field from its
first molecular solve, the oracle worker installs a read-only capture wrapper
around the two exact `solve_molecular_equilibrium` invocations. It records
arguments, dtype/device, returned arrays, iterations, and call order without
changing results. An independent replay may supplement this trace but cannot
be mislabeled as the hidden first call.

The reader calculation is complete before a golden is opened. The comparison
harness:

- requires exact numeric key-set equality;
- checks shape, dtype, axis metadata, and finiteness first;
- rejects missing, unexpected, NaN, and infinite fields;
- key-aligns catalogs by molecule code;
- reports maximum absolute, scale-relative, resolved-relative, and zero-leakage
  metrics;
- reports physical residuals separately from backend value differences;
- treats iteration exhaustion as an acceptance failure even though the exact
  solver returns a last state.

CPU source-staged paths target identity wherever operation order is unchanged.
A decomposed teaching helper receives a tolerance only after a measured
comparison to the exact staged path. CUDA/MPS thresholds are separately
measured; no Chapter 3 tolerance is copied by analogy.

## 14. Progressive source staging

Stage in dependency order:

1. Verify the two inherited Chapter 3 synthesis assets, then copy and manifest
   the three complete new molecular assets and the exact continuum-edge asset
   required by the public structured builder.
2. Stage exact `payne_zero_atmosphere.molecular_data`.
3. Stage exact `payne_zero_atmosphere.molecular_equilibrium` in full.
4. Extend the progressive atmosphere runner with exact
   `prepare_population_state` while retaining the already exact structured
   handoff.
5. Stage exact `payne_zero_synthesis.molecular_equilibrium` in full.
6. Exercise the already staged full synthesis EOS molecule branches.
7. Pin the exact synthesis partition-policy, hard-coded molecular-mass
   authority, host line-population normalization, continuum-edge loader, and
   public structured-builder mapping/reuse fragments without staging the
   Chapter 8 molecular opacity machinery.
8. AST/hash-verify public and private residual, update, continuation, and lane
   boundaries.
9. Apply the narrow Chapter 4 primary/support overrides from
   `design/chapter04_ownership_data_audit.md` to
   `audit/paynezero_symbol_coverage.json`; do not move whole mixed-ownership
   modules.

Add tests that fail if:

- either Newton unknown becomes a log density;
- either depth solve becomes batched or reordered;
- synthesis `vmap` moves inside a depth loop;
- the two backend positivity policies are merged;
- active and padded catalog extents are confused;
- goldens influence candidate construction.

Catalog tests also require the atmosphere coefficient row 6 to be zero and
name every genuinely zero-padded region: inactive molecule-code and
coefficient columns, post-active component starts and component indices, and
post-active equation-species entries. The atmosphere
`species_to_equation_index` reverse lookup is different: its 79 unused entries
are `-1`, code 100 maps to the electron row, and code 101 maps to the
one-past-active inverse-electron sentinel. Tests also pin component bounds, all
14 negative-ion ordering invariants, and malformed controlled copies that
expose the canonical NPZ loader's lack of schema validation.

## 15. Three movements and 17 visible cells

There are no detached exercises and no source-file dumps in Markdown.

### Movement I — One molecule couples two budgets

1. Asset and fixture preflight: hashes, counts, shapes, units, dtype.
2. One positive \(A+B\rightleftharpoons AB\) mass-action example with predicted
   temperature/pressure directions.
3. Decode real CO code 608 into C and O equation rows.
4. One professional single-claim H2/formation-policy plot with exact gates;
   interpret raw constants separately from normalized line populations.
5. Build a readable two-element, one-molecule residual and show why separate
   element solves fail.

### Movement II — Keep one coupled root positive through depth

6. One-depth residual and analytic atmosphere Jacobian; compare with finite
   differences.
7. Execute the atmosphere's unscaled `np.linalg.solve` step and forced
   `lstsq` fallback.
8. Feed controlled updates through the atmosphere reflection, sign-flip, and
   shared-scale branches only.
9. Run independent atmosphere warm/cool/high-pressure controls and interpret
   residuals, conservation, iterations, selected raw populations, transformed
   equation columns, and normalized populations across exact gates.
10. Run the ordered six-depth atmosphere track and compare deliberate restarts
    in one one-claim plot.
11. Save/restore physical equation densities, demonstrate the energy-mode seed
    override and early return, and evaluate one molecular-energy checkpoint.

### Movement III — Cross the atmosphere/synthesis boundary honestly

12. Execute one complete synthesis Newton step: log-product residual,
    finite-difference-versus-`jacrev` check, density-column scaling, and the
    synthesis-only positive-floor update.
13. Execute the synthesis full route; expose published/internal electron
    columns, shapes, residuals, selected molecules, and parity.
14. Execute the fixed-public-\(n_e\) route; prove exact public preservation and
    show why it is not internal charge closure.
15. Execute the route-claim matrix, including atmosphere handoff \(n_e\)
    movement, the pressure-iteration-disabled route, and global H2 bridge
    selection.
16. Key-align 170 shared catalog records and identify 20 synthesis-only codes.
17. Map CO to `[depth,5,45]`, verify all 54 synthetic lanes, and print the
    exact Chapter 5 handoff fields.

Each code cell states reads, writes, units, axes, dtype/device, and physical
question. Visible cells remain at most 35 lines and 92 characters per line.
Each result is interpreted before the next abstraction appears.

## 16. Visual contract

Original schematics:

1. separate C and O budgets become one coupled network through CO;
2. residual → Jacobian → density update, with distinct atmosphere and
   synthesis positivity branches;
3. ordered depth continuation in both backends, with `jacrev` inside one
   synthesis depth and `vmap` only after every depth loop has terminated;
4. atmosphere 170-state, synthesis 190-state, and public synthetic-lane
   boundary.

Every schematic has an owned prompt, provenance record, SHA-256, alt text,
caption, and scientific audit. Website art is an aesthetic reference only.

Quantitative plots are one panel and one physical claim. Use the book's
paper-inspired typography and restrained palette. Do not use multi-panel
figures to compress unresolved explanation.

## 17. Acceptance gates

Chapter 4 cannot be accepted until:

1. all three molecular assets, the continuum-edge asset, and two inherited
   synthesis assets are identity-bound, manifested, and shape/count checked;
2. atmosphere shapes are `(D,170)` and `(D,23)`;
3. synthesis shapes are padded `(D,200)` and `(D,30)` with exact zeros;
4. independent temperature/pressure controls follow the predicted directions;
5. analytic atmosphere and synthesis `jacrev` Jacobians pass independent
   finite-difference checks;
6. exact `0.69`, one-percent floor, atmosphere reflection/shared scale, and
   synthesis positive-floor branches execute visibly;
7. production continuation and deliberate resets are labeled separately;
8. no backend is described as a depth-batched molecular solve;
9. every state is finite and has route-appropriate conservation/residual
   reports;
10. atmosphere molecule-enabled handoff retains both input and output \(n_e\)
    and makes no preservation claim;
11. synthesis fixed route preserves public \(n_e\) exactly and makes no
    internal charge-closure claim;
12. the synthesis full route retains evidence from both molecular solves;
13. catalogs are compared by code and semantics, not row;
14. all 54 molecular public mappings are verified and only the normalized
    synthetic lane receives molecular line values, including a
    ground-floor-on/off normalization checkpoint;
15. the live/debug shape incompatibility is reproduced or explicitly
    source-probed, while release parity is tested against the synthesis
    product path;
16. molecular specific internal energy is taught once, with convection
    orchestration deferred;
17. all five golden archives are comparison-only and identity-bound;
18. 15–18 visible code cells remain bite-sized, with no detached exercise set
    and no Markdown source dump;
19. Chapter 3→4 and Chapter 4→5 continuity, notation, redundancy, visual, and
    whole-book audits pass.
20. synthesis seed, mixed-dtype full-route calls, MPS policy, and hidden
    nonfinite-residual replacement are explicitly tested;
21. atmosphere frozen-constant, post-solve recomputation, charge-square
    lifecycle, pressure-iteration-disabled, padding, component-bound, and
    negative-ion ordering gates pass;
22. the chapter wrapper accepts only `(99,)` abundances and validates positive
    finite thermodynamic columns plus intended depth order;
23. the symbol-coverage ledger contains the reviewed narrow Chapter 4
    overrides.
24. atmosphere raw, transformed-equation, and partition-normalized molecular
    arrays are independently tested across the 10000 K and 20000 K gates;
25. population modes 2/12 and specific-energy mode are proved to return before
    normalized filling, and energy mode neither overwrites saved physical
    warm starts nor silently uses the ordinary continuation seed;
26. the live bridge's mixed positive/zero H2 vector and all-zero analytic
    fallback follow the exact global selection rule;
27. zero-padding assertions exclude the atmosphere reverse lookup, whose
    `-1`, electron-row, and one-past-active sentinel semantics are pinned;
28. controlled diagnostics expose guarded nonfinite H2 constants versus
    unguarded ordinary-polynomial exponent overflow without feeding malformed
    data to a production golden.
29. synthesis cold-chain seeds, the per-network `1e-20` abundance floor, the
    direct omitted-dtype `float32` default, and explicit dtype on every
    scientific direct call are pinned;
30. the chapter wrapper rejects nonfinite or negative abundances and a
    nonpositive abundance sum without renormalizing valid inputs;
31. the public molecular-lines structured builder loads the exact edge grid,
    performs one fixed molecular solve, reuses its arrays, and does not enter
    the fallback re-solve;
32. the hard-coded `_ATOMIC_MASSES_FOR_MOLECULES` identity and at least one
    line-population normalization value are pinned independently of
    `atomic_masses.npz`.
33. molecular thermal-energy examples either use `run_atmosphere_model` or
    pass the tracking flag explicitly, and a source test pins the public-true
    versus lower-level-false default asymmetry.
34. atmosphere Doppler support preserves exact structural `+inf` values at
    zero-based packed slots 919 and 927, where the major-isotope mass and both
    actual and partition-normalized populations are exactly zero; every other
    Doppler slot must be finite.

## 18. Closing claim

The closing summary must establish:

- molecules turn independent elemental ledgers into one coupled density-space
  root;
- both production backends preserve ordered depth continuation;
- log evaluation belongs only to synthesis molecular products;
- atmosphere and synthesis positivity policies are intentionally distinct;
- full, fixed, diagnostic, and release routes own different electron-density
  and state claims;
- the two catalogs and three H2 policies are exact, distinct inputs;
- Chapter 5 can now ask how the resulting particles absorb and scatter
  continuum photons.

The chapter ends with an explicit link to Chapter 5.
