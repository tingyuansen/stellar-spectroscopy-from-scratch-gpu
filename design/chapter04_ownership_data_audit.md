# Chapter 4 Ownership, Data, and Parity Audit

Audit date: 2026-07-30

This is a bounded pre-draft audit for Chapter 4, **Molecules and Coupled
Equilibrium**. It reads the pinned Payne Zero checkout at commit
`9c44001feae40b85146630499e6f8a5fed42e5af` as a read-only oracle. It does not
authorize edits to that checkout, the paper, canonical chapters, source
staging, data, or `audit/paynezero_symbol_coverage.json`.

The chapter has one causal question:

> Once several elements may bind into the same molecule, how can mass action,
> elemental conservation, particle conservation, and charge determine one
> positive thermochemical state at every depth?

Chapter 3 has already earned atomic partition functions, Saha ionization,
charge closure, the atmosphere packed layout, the public synthesis cube, and
the distinction between actual and partition-normalized populations. Chapter
4 must consume those definitions without re-deriving them. Chapter 5 may use
the resulting H, H2, CH, OH, and electron populations in continuum opacity, but
Chapter 4 must not preview opacity formulas.

## Outcome

The fifteen-chapter architecture remains sound. Chapter 4 fits as one
three-movement navigation chapter with 17 visible code cells if the exact
full/fixed route boundaries below are respected.

There is one P0 claim blocker and four P1 contract issues to settle before
canonical prose is drafted:

1. **P0 — the molecule-enabled atmosphere handoff is not a fixed-electron-
   density calculation.** Its docstring is safe only for the atom-only branch.
2. **P1 — the synthesis “full” object combines the electron result of one
   molecular solve with molecular arrays from a second solve.** These values
   are normally close, but equality is not an interface invariant.
3. **P1 — public axis index 5 is overloaded.** It is a physical sixth ion-stage
   slot for atomic columns and a synthetic molecular-population lane for 54
   molecular species columns.
4. **P1 — the two molecular catalogs and the H2 treatments are deliberately
   different.** Whole-array or row-index parity across engines would be a false
   check.
5. **P1 — the current symbol-coverage JSON leaves several molecular objects
   under module defaults owned by Chapters 3, 7, 11, or Appendix C.** Narrow
   symbol overrides are required; moving whole modules would create new
   errors.

The exact density-space Newton description in
`design/global_chapter_contracts.md` is correct. Neither backend solves for
log-density unknowns. Only the synthesis molecular products are evaluated in
log space.

## Pinned source identities

| Source | SHA-256 | Chapter 4 responsibility |
| --- | --- | --- |
| `payne_zero_atmosphere/molecular_data.py` | `705c3072d79c8019c948ce0fa2c82052f232816d453e10a7c8e5fc5a8f5ce249` | atmosphere catalog schema, fixed-width provenance parser, default path |
| `payne_zero_atmosphere/molecular_equilibrium.py` | `4c9665148a57fadf4837f193c9f4247fbab44b788e7fb76ef3cae7ebdb3c3d86` | atmosphere formation constants, analytic residual/Jacobian, compiled update, continuation, line normalization, molecular energy |
| `payne_zero_atmosphere/equation_of_state.py` | `719b316327fd6fb76dfa5267a2c9022d1384c269a0fa20d5b2ec733671ae3fa2` | molecule/atom dispatch and the sentinel solve call |
| `payne_zero_atmosphere/runner.py` | `05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7` | full population route and structured-handoff ordering |
| `payne_zero_atmosphere/synthesis_bridge.py` | `142a960b5e710823754b02766803b3c1dd8c48c9945fdfabe560b4ee7e1acb50` | packed-to-public mapping and synthetic molecular lane |
| `payne_zero_synthesis/molecular_equilibrium.py` | `df01757c160b2bff4390cc2148cff9d1ba6e5a2bc7cab4515b46f38e868d2714` | synthesis catalog, log-evaluated products, `jacrev` solve, continuation, line normalization |
| `payne_zero_synthesis/equation_of_state.py` | `6497c29abb954e0b55d918cc22fa7b660952812c548faf1d7b1053345ef13562` | molecular seeds, full/fixed population routes, molecule-backed packed state |
| `payne_zero_synthesis/pipeline.py` | `465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22` | fixed-\(n_e\) structured bridge and synthetic molecular slot fill |
| `payne_zero_synthesis/atmosphere_schema.json` | `2ba8d637e613be12ff43ce319a752616323f0341ea69f8e2391c3c244939777a` | public `(depth, 6, 139)` field contract |

Line numbers below refer to these exact files at the pinned commit.

## Priority findings

### P0. Molecule-enabled `prepare_structured_handoff_population_state` does not preserve its input electron density

The atmosphere handoff docstring says “at fixed final electron density”
(`runner.py:288-295`). The molecule-enabled branch nevertheless does the
following:

1. initializes a molecular state (`runner.py:306-328`);
2. calls `populate_species(code=0.0, pressure_iteration_enabled=True)`
   (`runner.py:329-341`);
3. that sentinel call invokes `solve_molecular_equilibrium`
   (`equation_of_state.py:1577-1590`);
4. the molecular solve overwrites `total_nuclei_number_density`,
   `mass_density`, and `electron_density`
   (`molecular_equilibrium.py:296-313`);
5. only the subsequent packed population refill is fixed, through
   `pressure_iteration_enabled=False` (`runner.py:343-351`).

Therefore:

- the **atom-only** handoff from Chapter 3 preserves supplied \(n_e\);
- the **molecule-enabled** handoff runs one coupled molecular solve before its
  fixed atomic refill;
- the latter must not be labeled “the fixed-\(n_e\) atmosphere route.”

A direct pinned-source probe makes the distinction material. On the controlled
six-depth track

```text
temperature K:       [9000, 7000, 5500, 4500, 3800, 3300]
gas pressure cgs:    [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
```

feeding the molecule-enabled `prepare_population_state` electron-density
result back into `prepare_structured_handoff_population_state` changed the six
values by relative amounts

```text
[2.02e-3, 1.84e-3, 1.06e-2, 4.45e-2, 1.77e-1, 6.36e-1].
```

This probe is deliberately a thermochemical stress track, not a converged
stellar atmosphere. It shows that exact preservation cannot be asserted from
the source ordering alone.

**Required resolution.** The Chapter 4 claim table and tests must call this
route “molecule-enabled coupled solve followed by a fixed packed refill.”
Record both input and output \(n_e\) in its golden. Do not weaken the exact
atom-only fixed-\(n_e\) claim already established in Chapter 3.

### P1. Synthesis full-state fields can come from two molecular solves

`solve_population_state(..., molecules=True)` first calls
`solve_electron_density` (`equation_of_state.py:2045-2070`). After the atomic
fixed point, `solve_electron_density` runs molecular equilibrium, adopts the
returned electron and nuclei densities, and recomputes the EOS
(`equation_of_state.py:1594-1633`).

Assembly then calls `_molecule_backed_population_state`
(`equation_of_state.py:2198-2219`), which constructs fresh ion-formation
constants and runs molecular equilibrium again
(`equation_of_state.py:1850-1897`). The final `PopulationState` takes:

- `electron_density` from the first `ElectronDensityResult`;
- `molecular_populations`, `molecular_equation_densities`, and molecular
  `total_nuclei_number_density` from the second solve.

On the same stress track, the second solve's electron equation and the
published full-state `electron_density` differed by up to
`6.64e-4` relatively. The route is still the exact production route; the
unsafe claim is that every returned field is one indivisible root from one
solve.

**Required resolution.** Goldens must retain both
`population_state.electron_density` and the active electron column of
`molecular_equation_densities`. Conservation and residual checks must be
evaluated against the state from which each field came. The text should show
the two calls explicitly once, then use the exact route name without repeating
the implementation seam.

### P1. Public population axis index 5 has two meanings

The schema declares both population cubes as `(n_depth, 6, 139)`. For ordinary
atomic columns `0:99`, index 5 can hold a real sixth ion stage. Molecular bands
instead read:

```python
molecular_population_slot = partition_normalized_populations[:, 5, :]
```

(`molecular_lines.py:435-460`). The atmosphere bridge and synthesis pipeline
fill this lane by

```python
population_column = species_code // 6 - 1
partition_normalized_populations[:, 5, population_column] = population
```

(`synthesis_bridge.py:316-340`; `pipeline.py:564-581`).

There are exactly 54 supported molecular line-list species codes, from 240 to
792, mapped to 54 distinct public columns from 39 to 131. Those assignments
can overwrite what a universal “sixth atomic stage” interpretation would
expect in the partition-normalized cube. They do **not** turn
`ion_stage_populations[:, 5, :]` into molecular number densities, and molecular
Doppler widths are later calculated from molecular masses rather than borrowed
from `fractional_doppler_widths[:, 5, :]`.

**Required resolution.** Chapter 4 must amend, not contradict, Chapter 3:
the first 99 columns and first five stage indices retain the atomic layout;
index 5 is a context-dependent production lane. Show one concrete mapping,
for example species code 276 \(\rightarrow\) molecule code 608 (CO)
\(\rightarrow\) public column \(276//6-1=45\). Chapter 8 may then consume this
lane without re-explaining it.

### P1. Catalog identity and row identity are different claims

The atmosphere and synthesis catalogs share:

- the same 23 active equation species
  `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 19, 20, 22, 23, 24, 26, 100]`;
- 170 molecule codes with semantically identical coefficient and component
  records when key-aligned by molecule code.

They do not share row order. The synthesis catalog contains 20 additional
codes:

```text
111, 10811, 10812, 10820, 60606, 60608, 60614, 60816, 61414, 61616,
70708, 70808, 80814, 80816, 1010106, 1010107, 1010606, 6060707,
101010106, 101010114.
```

Whole-array equality or “first 170 rows” equality is therefore false even for
the shared physical records.

**Required resolution.** Teach code-key alignment as the scientific
comparison. Golden metadata records catalog identity separately from
code-aligned value comparisons. Never sort or rewrite the production catalogs
themselves.

### P1. H2 has three exact policies, not one universal helper

The atmosphere molecular solver reads the 200-value
`h2_partition_function`, clamps interpolation to 100–19900 K, and uses
`36118.11 cm^-1` in its special code-101 equilibrium constant
(`molecular_equilibrium.py:71-139`, `641-724`). The special atmosphere
equilibrium branch is allowed through 20000 K.

The synthesis molecular solver does not read that partition table. Its
code-101 branch evaluates a hard-coded polynomial containing the 4.478 eV
dissociation scale and zeros all polynomial-molecule formation constants above
10000 K
(`molecular_equilibrium.py:181-281`). The local boolean is named `is_hminus`,
but the surrounding catalog, pipeline, and output mapping use molecule code
101 as molecular hydrogen; the textbook must explain the physical role and
retain the exact source spelling only where quoting the production boundary.

Finally, the synthesis structured pipeline forms its named
`molecular_hydrogen_population` with the 4.478 eV polynomial and sets it to
zero above 9000 K (`pipeline.py:403-431`), replacing that provisional value
with the solved code-101 population when molecules are active
(`pipeline.py:550-562`).

**Required resolution.** Compare the three policies in one compact table.
Do not use the atmosphere H2 partition asset as a synthesis input, and do not
claim atmosphere/synthesis H2 equality outside an explicitly measured route.

## Symbol-ownership corrections

The JSON uses module defaults until a narrow override is reviewed. The
following are the required overrides. Appendix C can remain a supporting API
inventory, but it is not the primary pedagogical home of physical routines.

### Definite Chapter 4 primary overrides

| Currently mapped object(s) | Current primary | Correct primary/support |
| --- | --- | --- |
| `payne_zero_atmosphere.__init__.MolecularEquilibriumCatalog`, `.MolecularEquilibriumState`, `.compute_equilibrium_constants_for_layer`, `.compute_molecular_specific_internal_energy`, `.initialize_molecular_equilibrium_state`, `.find_default_molecular_equilibrium_catalog`, `.parse_molecular_equilibrium_record`, `.populate_molecular_species`, `.read_molecular_equilibrium_catalog`, `.restore_molecular_equation_density`, `.save_molecular_equation_density`, `.set_molecular_specific_internal_energy_mode`, `.solve_molecular_equilibrium`, `.solve_molecular_equilibrium_layer` | Appendix C | Chapter 4 primary; Appendix C supporting |
| `payne_zero_synthesis.equation_of_state.molecular_seed_electron_density` and `.molecular_ion_formation_constants_from_seed` | Chapter 3 | Chapter 4 primary; Chapter 3 only supplies the atomic Saha state they consume |
| `payne_zero_synthesis.equation_of_state.PopulationState.molecular_populations` and `.molecular_equation_densities` | Chapter 3 | Chapter 4 primary; Chapter 10 consumes |
| `payne_zero_atmosphere.runner.AtmospherePopulationState.molecular_state` | Chapter 11 | Chapter 4 primary; Chapters 11–12 consume |
| `payne_zero_atmosphere.source_catalogs.molecular_equilibrium_catalog_path` | Chapter 2 | Chapter 4 primary; Chapter 2 generic path/checksum discipline and Appendix B installation support |
| `payne_zero_atmosphere.config.AtmosphereInput.molecules_path`, `AtmosphereConfig.enable_molecules`, and `run_setup.RunSetup.molecules_enabled` | Chapter 11 | Chapter 4 primary as the chemistry gate; Chapter 11 supports full runner integration |
| `payne_zero_synthesis.pipeline.WindowInvariants.molecular_lines`, `.molecular_invariants`, and `.n_molecular` | Chapter 7 | Chapter 8 primary; Chapter 10 supporting |
| `payne_zero_atmosphere.config.AtmosphereConfig.molecular_convection_thermal_tracks_perturbation` | Chapter 11 | Chapter 12 primary; Chapters 11 and 13 supporting |

The remaining molecular-looking Appendix C exports also need origin-specific
overrides rather than a Chapter 4 override:

| Current Appendix C export(s) | Correct primary/support |
| --- | --- |
| `payne_zero_atmosphere.__init__.MolecularEquilibriumTables`, `.load_molecular_equilibrium_tables`, `.compute_molecular_hydrogen_population`, `.compute_molecular_hydrogen_ion_opacity_columns`, `.compute_molecular_continuum_opacity_columns` | Chapter 5 primary; Appendix C supporting |
| `payne_zero_atmosphere.__init__.compute_hydrogen_molecule_population`, `.molecular_hydrogen_equilibrium_constant` | Chapter 7 primary; Appendix C supporting |

The full functions `solve_electron_density`, `solve_population_state`,
`solve_population_state_at_electron_density`, `populate_species`, and
`populate_all_species` remain Chapter 3 primary because their atom-only
contract was earned there. Chapter 4 owns and tests their molecule-enabled
branches as a supporting location; moving the entire functions would create
redundancy.

### Names that look molecular but are already correctly owned elsewhere

- `continuum_opacity.MolecularEquilibriumTables`,
  `load_molecular_equilibrium_tables`,
  `compute_molecular_hydrogen_population`,
  `compute_molecular_hydrogen_ion_opacity_columns`, collision-table fields, and
  `compute_molecular_continuum_opacity_columns` stay in Chapter 5.
- `hydrogen_line_profile.molecular_hydrogen_equilibrium_constant`,
  `compute_hydrogen_molecule_population`, the quasi-molecular cutoff fields,
  and the evaluator's `molecular_hydrogen_population` stay in Chapter 7.
- All `molecular_lines` catalog, invariant, mass, chunk, Doppler, and
  accumulation objects and all molecular source compilers stay in Chapter 8.
- Their `__init__` re-exports should inherit those physical chapters as
  primary and Appendix C as supporting; they should not be moved into Chapter
  4 merely because “molecular” appears in the name.

## Exact data contract

These three complete assets are sufficient for the Chapter 4 chemistry. They
are small enough to copy into the textbook without an installer.

| Upstream asset | Bytes | SHA-256 | Exact contents | Textbook role |
| --- | ---: | --- | --- | --- |
| `source_data_files/source_catalogs/lines/molecular_equilibrium_atmosphere.npz` | 19,040 | `971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644` | active counts 170 molecules, 23 equations, 481 components; buffers `molecule_codes (200,) f8`, `equilibrium_coefficients (7,200) f8`, `component_start_indices (201,) i4`, `component_equation_indices (600,) i4`, `equation_species_codes (35,) i4`, `species_to_equation_index (102,) i4` | `data/static`; complete atmosphere chemistry catalog |
| `source_data_files/source_catalogs/lines/molecular_equilibrium_synthesis.npz` | 18,060 | `3e8c1ea69fe672b9886bda38922f868c6d2ac2b43c4eb0d7750620241c238d28` | active counts 190 molecules, 23 equations, 548 components; same 200/600 fixed buffers, with `equation_species_codes (30,) i4` and no reverse lookup field | `data/static`; complete synthesis chemistry catalog |
| `source_data_files/atmosphere_tables/molecular_equilibrium_tables.npz` | 1,935 | `1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe` | `h2_partition_function (200,) f8`, range 0.667–663.556; `atomic_mass_amu (99,) f8` | `data/static`; atmosphere H2 partition table, shared later by Chapters 5 and 7 |

Recommended self-contained destinations preserve the runtime-relative paths:

```text
data/static/source_catalogs/lines/molecular_equilibrium_atmosphere.npz
data/static/source_catalogs/lines/molecular_equilibrium_synthesis.npz
data/static/atmosphere_tables/molecular_equilibrium_tables.npz
```

The `atomic_mass_amu` member of the last archive is not read by the Chapter 4
atmosphere molecular module; it must not become a new mass authority. Keep the
whole byte-identical archive because Chapters 5 and 7 resolve the same file.

These files are not `data/subsets`: they are the complete runtime catalogs and
table. The proposed controlled thermochemical columns belong in
`data/fixtures`; pinned outputs belong in `data/golden/payne_zero` and may be
opened only after the reader calculation.

## Exact numerical contract

### Molecular codes and the coupled unknown

The molecule catalogs encode each molecule as base-100 element identifiers,
plus inverse-electron sentinels for positive ion charge. Decoding produces:

- a molecule code;
- seven equilibrium coefficients;
- a ragged component list represented by start indices and equation indices;
- one ordered density unknown per active element equation, plus the total
  nuclei scale and, when present, an electron equation.

For molecule \(m\), the mass-action term has the form

\[
n_m = K_m(T)\left(\prod_j x_j^{\nu_{mj}}\right)n_e^{-p_m},
\]

where \(x_j\) are **density-space** unknowns, \(\nu_{mj}\) are component
multiplicities, and \(p_m\) counts inverse-electron sentinels.

Element budgets are coupled because one \(n_m\) contributes to every element
that molecule contains. The total-particle row and electron/negative-ion
corrections close the remaining equations.

### Atmosphere update

The atmosphere implementation:

- constructs formation constants in an exact ordered molecule loop;
- assembles an analytic residual and Jacobian in compiled Numba kernels;
- uses `np.linalg.solve`, with `np.linalg.lstsq` only after
  `LinAlgError`;
- tests the relative update against `1e-4` **before** damping;
- multiplies a sign-flipping update by `0.69`;
- accepts `abs(current - delta)` when its magnitude is at least one percent of
  the current density;
- otherwise floors to `current/scale`, where `scale` starts at 100 and becomes
  `sqrt(scale)` for later equations if the same update also flipped sign
  (`molecular_equilibrium.py:861-885`);
- advances depths sequentially, scaling the preceding returned density vector
  by the gas-pressure ratio (`molecular_equilibrium.py:279-300`);
- never uses `prange` for this continuation.

The absolute-value recovery and the mutable, equation-ordered `scale` are
atmosphere-specific. They must not be silently replaced by the synthesis
policy.

### Synthesis update

The synthesis implementation:

- constructs the same kind of density-space unknown;
- evaluates molecular products through finite, dtype-safe logarithms
  (`molecular_equilibrium.py:356-419`);
- uses `torch.func.jacrev` on one depth's actual residual;
- column-scales the Jacobian by current density before
  `torch.linalg.solve`;
- tests convergence before damping;
- multiplies a sign-flipping update by `0.69`;
- replaces a candidate below `current/100` with `current/100`, without the
  atmosphere absolute-value recovery or cross-equation `sqrt(scale)` policy;
- advances depths in a Python loop, carrying the previous detached solution
  after pressure scaling;
- uses `chain_length` only to force explicit reset boundaries;
- applies `vmap` only after every depth loop terminates to evaluate final molecular densities
  (`molecular_equilibrium.py:545-651`).

The direct molecular solver default is `tol=1e-3`; both EOS population wrappers
use the production molecule-backed state at `1e-4`. A visible call must state
which boundary supplies the tolerance.

### Molecular specific internal energy

The atmosphere energy route is a second, explicitly enabled mode. A normal
population solve saves `previous_molecular_equation_densities`; later
temperature/pressure perturbations restore that warm state, set
`specific_internal_energy_mode_enabled`, and evaluate molecular contributions
with \(\pm0.1\%\) temperature samples
(`molecular_equilibrium.py:289-339`, `596-620`, `1212-1366`). Chapter 4 owns
the meaning and one checked value. Chapter 12 owns the four perturbed
convection calls and must not re-derive the molecular energy expression.

## Full/fixed molecule-enabled claim matrix

| Route | What happens to supplied \(n_e\)? | Molecular solve and state source | Mass/nuclei behavior | Safe reader-facing claim |
| --- | --- | --- | --- | --- |
| Atmosphere `prepare_population_state`, molecules on | seed only; coupled solver overwrites it | one ordered atmosphere molecular solve, followed by packed fills using that state | molecular solve writes `total_nuclei_number_density` and `mass_density` | full molecule-enabled atmosphere population phase |
| Atmosphere `prepare_structured_handoff_population_state`, molecules on | **not preserved**; coupled solver can overwrite it | one ordered molecular solve, then packed fills with `pressure_iteration_enabled=False` | molecular solve writes nuclei and mass | coupled molecular solve followed by fixed packed refill; not a fixed-\(n_e\) route |
| Synthesis `solve_population_state`, molecules on | atomic fixed point, then first molecular solve supplies published \(n_e\) | a second molecular solve supplies molecular arrays and molecule-backed packed populations | published nuclei come from second solve; mass is rescaled to that nuclei density | production full synthesis route, with a documented two-solve seam |
| Synthesis `solve_population_state_at_electron_density`, molecules on | `PopulationState.electron_density` preserves the supplied array exactly | internal molecular solve still carries an electron equation; its returned electron is ignored by the outer state | molecular nuclei replace ideal-gas nuclei; supplied mass is preserved, otherwise mass is molecule-rescaled | fixed **public** \(n_e\); no charge-closure claim |
| Synthesis pipeline structured bridge | calls the preceding fixed public-\(n_e\) route | reuses molecular arrays when present, otherwise solves them, then fills the synthetic lane | carries the fixed-route mass policy | synthesis-ready structure at an upstream-owned \(n_e\) |

The chapter must print this matrix after the two routes have been executed, not
use it as a substitute for executing them.

## Fixture design

Create one deterministic
`data/fixtures/chapter04_molecular_inputs.npz`. It should contain no output from
either backend.

### Shared mixture and bookkeeping

- Copy the exact 99-element linear abundance vector already used by Chapter 3
  so the chemistry changes, not the mixture definition.
- Store its source fixture SHA-256
  `3ed0d65431fc9e284a77011b82267241b25cc56cdffa73e1bc86eec15f9b5219`
  in the manifest, not as a runtime dependency.
- Store monotonically increasing `column_mass`, positive microturbulence, a
  declared electron-density **seed**, and the exact outer-to-inner order.
- Call every array a controlled thermochemical fixture, never a converged
  atmosphere.

### Main six-depth continuation track

Use a physically readable outer-to-inner trend:

```text
temperature K:       [3300, 3800, 4500, 5500, 7000, 9000]
gas pressure cgs:    [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
column mass:         geomspace(1e-6, 1e2, 6)
microturbulence:     2e5 cm s^-1 at every depth
electron seed:       0.1 * gas_pressure / (k_B,reference * temperature)
```

This covers a cool outer layer, molecule-rich intermediate layers, and a warm
high-pressure deep layer while retaining the global depth direction. It is the
ordered-continuation and route-integration fixture.

### Isolated limiting controls

Store two additional three-point controls:

```text
temperature control:
    T = [3500, 6000, 9000] K
    P = [1e5, 1e5, 1e5] dyn cm^-2

pressure control:
    T = [3500, 3500, 3500] K
    P = [1e2, 1e4, 1e6] dyn cm^-2
```

Evaluate these as independent one-depth cases, or use synthesis
`chain_length=1`, so the warm/cool and pressure claims are not contaminated by
continuation. The main track separately demonstrates why resetting every depth
changes the numerical trajectory even when the accepted roots are close.

### Required named probes

Key-align and retain at least H2 `101`, C2 `606`, CN `607`, CO `608`, N2 `707`,
NO `708`, and O2 `808`. These provide one homonuclear molecule, the CNO network,
and a direct bridge to the Chapter 1 motivation without turning the chapter
into an abundance survey.

## Deterministic golden plan

All goldens are comparison-only and are generated in a fresh subprocess whose
first import resolves under `/Users/ysting/payne-zero`. Use:

- CPU only;
- NumPy and Torch `float64`;
- one Torch thread and one BLAS thread;
- one Numba thread, although molecular depth continuation itself is ordered;
- `PYTHONHASHSEED=0`;
- the repository's deterministic NPZ writer;
- exact fixture, data-asset, module, and pinned-commit identity fields.

Write five focused products rather than one opaque archive:

1. `chapter04_molecular_constants_cpu_float64.npz`
   - both catalog active metadata;
   - code-key alignment;
   - atmosphere H2 interpolation and equilibrium constants;
   - synthesis polynomial formation constants;
   - isolated temperature/pressure control results.
2. `chapter04_atmosphere_molecular_state_cpu_float64.npz`
   - full-route input seed and output electron density;
   - nuclei, mass, charge-square, packed actual and partition-normalized
     populations;
   - active molecule codes, molecular populations, partition-normalized
     molecular populations, equation densities, saved previous densities;
   - one explicitly enabled molecular specific-energy result;
   - molecule-enabled structured-handoff input/output \(n_e\) and their
     measured delta.
3. `chapter04_synthesis_molecular_full_cpu_float64.npz`
   - published electron/nuclei/mass fields;
   - the active electron equation from the second molecular solve;
   - padded `(D,200)` molecular populations and `(D,30)` equation densities;
   - public atomic population cubes and all named population fields.
4. `chapter04_synthesis_molecular_fixed_cpu_float64.npz`
   - supplied and published \(n_e\), required to be byte-equal;
   - internal electron equation, nuclei, mass, molecule arrays, and cubes;
   - both the supplied-mass and derived-mass branches.
5. `chapter04_molecular_public_mapping_cpu_float64.npz`
   - the exact 54 species-code/molecule-code/public-column triples;
   - atmosphere-bridge and synthesis-pipeline synthetic lane values;
   - the untouched actual-ion cube and a sentinel showing why index 5 cannot
     be interpreted uniformly.

The reader computes first, verifies finiteness and normalized residuals, and
only then loads the corresponding golden. Exact CPU source/data replays should
target array identity. CUDA/MPS comparisons require measured field-specific
tolerances and are not part of the immutable golden name.

The comparison harness must:

- require exact numeric key-set equality;
- reject missing fields and non-finite values;
- key-align catalogs by rounded molecule-code key, never row index;
- report maximum absolute, scale-relative, resolved-relative, and zero-leakage
  metrics;
- report residual norms and conservation defects separately from backend
  value differences;
- record ordered-continuation versus reset trajectories without claiming that
  the reset path is production.

## Progressive source-staging dependencies

Chapter 4 should extend the current Chapter 3 source stage in this order:

1. **Static inputs and codecs**
   - copy the three exact assets above;
   - stage exact `payne_zero_atmosphere.molecular_data`;
   - add all three assets and both catalog active-count checks to
     `PAYNE_ZERO_SOURCE_MANIFEST.json`.
2. **Atmosphere chemistry**
   - stage exact `payne_zero_atmosphere.molecular_equilibrium` in full; trimming
     its private compiled kernels would change the production path;
   - retain its dependencies on the already staged constants, data path,
     runtime state, and Chapter 3 Saha stack;
   - extend the progressive atmosphere runner with the exact molecular imports
     and exact `prepare_population_state`;
   - make the existing exact handoff's molecular branch executable without
     rewriting its misleading fixed-\(n_e\) docstring as a stronger claim.
3. **Synthesis chemistry**
   - stage exact `payne_zero_synthesis.molecular_equilibrium` in full;
   - retain its already staged constants, device, paths, and EOS dependencies;
   - exercise the existing exact molecule-enabled branches of
     `solve_population_state` and
     `solve_population_state_at_electron_density`.
4. **Public mapping**
   - use the already staged atmosphere `synthesis_bridge` for its exact lane
     fill;
   - keep the full synthesis pipeline deferred to Chapter 10, but pin the exact
     `pipeline.py:438-582` fragment and reproduce its few visible mapping lines
     in the Chapter 4 calculation;
   - do not stage Chapter 8 molecular line catalogs or opacity accumulation
     merely to prove the lane index.
5. **Verification**
   - AST/hash-verify every staged public definition and the private residual,
     update, continuation, and lane-fill fragments;
   - compare the source stage against fresh pinned-source goldens;
   - add a source test that fails if either backend is rewritten as a batched
     depth Newton solve or if `vmap` moves inside a depth loop.

This ordering keeps every staged module importable while avoiding a premature
copy of the entire synthesis pipeline.

## Three-movement, 17-cell chapter checklist

There are no detached exercises. Every limiting case and failure probe appears
where it answers the current causal question.

### Movement I — One molecule couples two element budgets

Remembered question: why can CO not be solved by a carbon-only or oxygen-only
Saha calculation?

1. **C1 — Declared reads/writes and exact asset preflight.** Print the three
   hashes, active counts, shapes, units, CPU dtype, and fixture identity.
2. **C2 — One-reaction mass action.** Compute \(n_{\rm AB}=K n_{\rm A}n_{\rm B}\)
   for one deliberately small positive case; predict and verify the warm
   dissociation and high-pressure formation directions.
3. **C3 — Decode a real molecule record.** Walk CO code 608 into C and O
   component equations; compare catalog buffers by names, not a raw dump.
4. **C4 — Exact formation policies.** Plot one panel of the H2 atmosphere table
   interpolation and synthesis polynomial over temperature, marking their
   branch limits without asserting equality.
5. **C5 — From one reaction to a residual vector.** Build a two-element,
   one-molecule conservation residual in readable density-space code and show
   why solving each element separately violates one budget.

Natural pause: mass action supplies molecule densities only after all free
element densities are known together.

### Movement II — Keep the coupled root positive and follow it through depth

Remembered question: how does one find a positive root when the equations span
many orders of magnitude?

6. **C6 — Residual and Jacobian at one depth.** Expose the exact atmosphere
   analytic assembly on a compact selected network; compare its Jacobian with
   finite differences.
7. **C7 — One density-scaled Newton step.** Show why column scaling changes
   conditioning but not the density-space unknown.
8. **C8 — Exact damping branches inline.** Feed controlled sign-flip,
   negative-candidate, and sub-one-percent updates through both backend
   policies and interpret their different outcomes.
9. **C9 — Exact atmosphere one-layer solve.** Run warm, cool, and cool
   high-pressure controls independently; print normalized residual,
   conservation defects, iterations, and selected molecule fractions.
10. **C10 — Ordered continuation.** Run the six-depth atmosphere track in
    production order and compare with deliberate per-depth resets in one
    single-claim plot of iteration count or normalized residual.
11. **C11 — Molecular thermodynamics.** Save/restore equation densities,
    enable molecular specific-energy mode, calculate one perturbation pair,
    and explain the value; defer convection use to Chapter 12.

Natural pause: the atmosphere now has a checked molecular state, but synthesis
uses a different catalog, solver, and public mapping.

### Movement III — Cross the exact atmosphere/synthesis boundary honestly

Remembered question: which values are truly shared when the two engines use
different catalogs and route semantics?

12. **C12 — Torch residual for one depth.** Evaluate the same density-space
    residual with log products and compare `jacrev` with finite differences;
    state the selected device/dtype.
13. **C13 — Exact synthesis full route.** Run CPU float64, print published and
    internal electron columns, nuclei/mass, padded shapes, selected molecules,
    residuals, and golden parity.
14. **C14 — Exact synthesis fixed public-\(n_e\) route.** Pass C13's published
    density unchanged, verify exact preservation, and show why that does not
    establish charge closure for the internal molecular electron equation.
15. **C15 — Execute the full/fixed claim matrix.** Include the molecule-enabled
    atmosphere handoff delta; make each check green only for the claim the
    route actually owns.
16. **C16 — Code-key-aligned catalog comparison.** Verify 170 shared semantic
    records and 20 synthesis-only records; compare selected physical outputs
    as measured quantities, not presumed cross-engine equality.
17. **C17 — Synthetic public lane.** Map CO's code to index
    `[depth, 5, 45]`, verify all 54 mappings and both exact fill paths, then
    print the next-chapter handoff fields.

The closing summary must say:

- molecules turn independent elemental accounting into one coupled
  density-space root;
- both production backends preserve ordered depth continuation;
- log evaluation belongs to synthesis molecular products, not the unknown;
- full and fixed routes own different electron-density claims;
- the molecular catalogs and H2 policies are distinct exact inputs;
- Chapter 5 may now ask how these checked populations absorb and scatter a
  photon continuously.

End with an explicit link to Chapter 5.

## Acceptance gates before Chapter 4 is complete

1. All three static assets are byte-identical to the pinned source and recorded
   in `data/MANIFEST.json`.
2. Atmosphere active shapes are exactly `(D,170)` molecular populations and
   `(D,23)` equation densities; synthesis outputs remain padded `(D,200)` and
   `(D,30)`.
3. The isolated temperature and pressure controls obey their predicted
   molecular trends without using a golden as input.
4. Analytic atmosphere and `jacrev` synthesis Jacobians each pass an
   independently computed finite-difference check.
5. The exact `0.69`, one-percent floor, atmosphere absolute recovery, and
   synthesis positive floor branches are all executed visibly.
6. Production continuation and deliberate resets are labeled and compared;
   neither backend is described as depth-batched.
7. Every returned state is finite, has a reported normalized residual, and
   satisfies the conservation claims appropriate to its route.
8. Atmosphere molecule-enabled handoff input/output \(n_e\) are both retained;
   no preservation assertion is made.
9. Synthesis fixed route preserves its public \(n_e\) exactly and makes no
   charge-closure claim.
10. Shared catalog records are compared by molecule code and component
    semantics; raw row equality is never used.
11. All 54 molecular species map to exact public columns and only
    `partition_normalized_populations[:,5,:]` receives the molecular line
    values.
12. Molecular specific internal energy is taught once here; convection
    perturbation orchestration is deferred to Chapter 12.
13. Chapter 5 receives named populations and densities but no molecular
    equilibrium re-derivation.
14. The chapter has 15–18 visible code cells, no detached exercise section,
    no source-file code dump in Markdown, and every plot has one physical
    claim.
