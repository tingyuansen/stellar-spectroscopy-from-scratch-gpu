# Chapter 4 pedagogical sequence: molecules and coupled equilibrium

Status: implementation-ready chapter contract  
Visible scientific code cells: **17**  
Movements: **3**  
Detached exercises: **none**

This document controls the rewrite of Chapter 4. It is subordinate to
`BIBLE.md`, `PLAN.md`, `design/global_chapter_contracts.md`, and
`design/chapter04_exact_source_contract.md`. Where a convenient teaching
shortcut would change a Payne Zero result, the exact source contract wins.

## Reader promise and chapter question

Chapter 3 ended with separate elemental ledgers. Every carbon nucleus belonged
to one carbon ion stage, every oxygen nucleus belonged to one oxygen ion stage,
and the electron density closed their combined charge. That description fails
as soon as one CO molecule owns a carbon nucleus and an oxygen nucleus at the
same time.

The chapter begins with that concrete failure:

> Given \(T\), \(P_{\rm gas}\), and the Chapter 3 elemental mixture, how can we
> assign positive atomic, ionic, molecular, and electron densities while
> conserving every elemental budget and the total particle budget?

The answer earned by the end is:

> Mass action predicts molecular populations for a trial set of free
> densities. Element, particle, and charge conservation then form one coupled
> density-space root. Both exact backends solve that root in depth order, but
> they own different positivity, precision, electron-density, catalog, and
> public-state contracts.

The reader is a final-year undergraduate or first-year graduate student who
knows basic calculus, linear algebra, and ideal-gas physics. The prose must
define every new physical object before naming its production array. It should
feel conversational and deliberate, not like an API tour.

## Global causal spine

The chapter has one forward motion:

```text
one molecule spends two budgets
        ↓
mass action alone is conditional on free densities
        ↓
simultaneous conservation defines a vector residual
        ↓
Newton needs a Jacobian, a linear solve, and a positivity policy
        ↓
the next depth inherits a pressure-scaled solution
        ↓
the atmosphere and synthesis backends preserve that physics differently
        ↓
their routes make different electron-density and state claims
        ↓
normalized molecular populations enter one explicit public synthetic lane
```

There is no source-file tour. A source name appears only when the physical
quantity or numerical operation it implements has already been earned.

## Inherited, new, and deferred notation

### Reused without re-derivation

The opening may remind the reader, in one sentence each, that:

- \(T\) is `temperature` in K and \(P_{\rm gas}\) is material
  `gas_pressure` in dyn cm\(^{-2}\);
- \(n_e\) is `electron_density`, and \(A_a\) is the Chapter 3 linear number
  abundance of element \(a\);
- \(n_{a,r}\), \(U_{a,r}\), and \(n_{a,r}/U_{a,r}\) retain their Chapter 3
  meanings;
- depth 0 is outermost.

The chapter must not re-teach Saha ionization, partition-function branches,
pressure lowering, abundance decoding, manifests, packed atomic schedules, or
atomic internal energy. It may call their exact implementations because the
current molecular iterate changes their numerical inputs.

### Earned in this chapter

Notation is introduced only in this order:

1. \(x_0\): the nuclei-density scale; \(x_a\): a free elemental density;
2. \(K_m(T)\): a formation constant; \(\nu_{ma}\): component multiplicity;
3. \(n_m(\mathbf{x})\): the mass-action population of catalog record \(m\);
4. \(\mathbf{R}(\mathbf{x})\), \(J_{ij}=\partial R_i/\partial x_j\), and
   \(J\boldsymbol{\delta}=\mathbf{R}\);
5. \(x_e\), inverse-electron power, and negative-ion sign corrections, only
   when the synthesis residual is reached.

All Newton unknowns are ordinary number densities in cm\(^{-3}\). The phrase
“log-space Newton” is forbidden. Only synthesis evaluates the molecular
product through logarithms.

### Deferred

Continuum cross-sections, line opacity, line profiles, radiative transfer,
convection theory, and spectrum formation are not Chapter 4 topics. The public
population lane is constructed here because Chapter 5 needs a trustworthy
state, not because opacity is being calculated here.

## Controlled inputs used throughout

The main six-depth track is fixed:

- `column_mass = geomspace(1e-6, 1e2, 6)` g cm\(^{-2}\);
- `temperature = [3300, 3800, 4500, 5500, 7000, 9000]` K;
- `gas_pressure = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]`
  dyn cm\(^{-2}\);
- `microturbulence = 2e5` cm s\(^{-1}\);
- the exact accepted Chapter 3 `(99,)` abundance vector;
- the declared Chapter 3 electron-density seed.

Independent direction checks do **not** reuse continuation:

- \(T=[3500,6000,9000]\) K at \(P_{\rm gas}=10^5\) dyn cm\(^{-2}\);
- \(P_{\rm gas}=[10^2,10^4,10^6]\) dyn cm\(^{-2}\) at \(T=3500\) K.

Exact branch probes are separate from plotting grids:

- 10000 K and `nextafter(10000, +inf)` for both polynomial policies;
- 9000 K and `nextafter(9000, +inf)` for provisional synthesis H2;
- 100, 101, 19899, 19900, 20000 K and
  `nextafter(20000, +inf)` for atmosphere H2.

Named species probes are H2 101, C2 606, CN 607, CO 608, N2 707, NO 708, and
O2 808. Hidden setup may import functions and load immutable arrays, but it
must not perform a scientific calculation later claimed by a visible cell.

# Movement I — One molecule couples two budgets

## Movement question

Chapter 3 could close each element's ion ladder separately once \(n_e\) was
known. Why can carbon and oxygen no longer be solved independently after CO is
allowed?

## Movement claim

Mass action tells us how much CO exists **for chosen free C and O densities**.
It does not choose those free densities. Because the same \(n_{\rm CO}\) term
appears in both elemental ledgers, the budgets must be solved together.

Place the first original schematic immediately before Cell 5, after the reader
has decoded CO but before the vector residual is introduced:

`assets/schematics/textbook/ch04-coupled-budgets-v1.png`

The image shows separate C and O ledgers on the left, one CO node in the
middle, and the same CO token subtracted from both ledgers on the right. It
must contain no equations beyond \(n_{\rm CO}\), no code, and no decorative
stellar photograph.

## Cell 1 — Bind the chemistry to exact bytes

- **Remembered question.** Chapter 2 taught that a familiar filename does not
  identify data. Which exact tables define this molecular problem?
- **New claim.** Catalog identity, active extent, padded extent, units, and
  dtype are part of the physical calculation; a matching hash proves byte
  identity, not physical correctness.
- **Minimal notation.** Only \(D\) for depth count, \(M\) for active catalog
  records, and \(E\) for active equation rows. No chemistry equation yet.
- **Exact code function/source shown.** Use the staged
  `payne_zero_atmosphere.molecular_data.read_molecular_equilibrium_catalog`
  and `payne_zero_synthesis.molecular_equilibrium.read_molecule_table`.
  Verify the local manifest and hashes for the two catalogs, the atmosphere H2
  table, `continuum_edge_grid.npz`, and the two inherited Chapter 3 synthesis
  assets. Do not print either source file or whole array.
- **Expected output.** One compact table with role, hash status, active
  shape/count, dtype, and units. It must show atmosphere `170 / 23 / 481`,
  synthesis `190 / 23 / 548`, the atmosphere H2 table `(200,)`, and the exact
  edge-grid identity. A controlled malformed copy demonstrates that the
  canonical atmosphere NPZ reader trusts its schema; the chapter wrapper
  rejects the malformed copy. Wrapper probes also reject nonpositive or
  nonfinite temperature/pressure, a depth-dependent abundance matrix, and
  reversed outer-to-inner order.
- **Immediate interpretation.** The two chemistry tables share equation
  width but are not interchangeable. Padding is storage capacity, not a
  chemical species count.
- **Contract earned.** Every later numerical claim is bound to exact immutable
  inputs. Valid abundances must be finite, nonnegative, shape `(99,)`, and have
  positive sum; they are not renormalized. This protective wrapper is
  deliberate: the low-level continuation does not validate depth order,
  molecule-backed synthesis silently takes the first row of a
  depth-dependent abundance matrix, and the atmosphere bridge rejects such
  variation.
- **Causal next link.** With the inputs identified, reduce the catalog to one
  transparent reaction before confronting its encoding.

## Cell 2 — Solve one positive \(A+B\rightleftharpoons AB\) balance

- **Remembered question.** What does a formation constant actually determine?
- **New claim.** For fixed \(x_0\), mass action and the two elemental budgets
  give one physical scalar root; increasing the nuclei-density scale favors a
  two-body molecule, while the temperature direction depends on the declared
  \(K(T)\).
- **Minimal notation.**
  \[
  n_{AB}=Kx_Ax_B,\quad
  x_A+n_{AB}=A_Ax_0,\quad
  x_B+n_{AB}=A_Bx_0.
  \]
  Define the molecular fraction relative to the limiting elemental budget.
- **Exact code function/source shown.** A readable, chapter-local
  `positive_ab_equilibrium` implements the analytic quadratic and selects the
  root satisfying
  \(0\le n_{AB}\le\min(A_A,A_B)x_0\). It is explicitly a teaching reduction,
  not a replacement Payne Zero API.
- **Expected output.** A four-row table for declared cool/hot and low/high
  \(x_0\) cases. It prints \(K\), \(n_{AB}\), limiting-budget fraction, and
  both conservation residuals. No plot is needed.
- **Immediate interpretation.** “Cooler makes more molecule” is justified
  only because this controlled pair declares \(K_{\rm cool}>K_{\rm hot}\);
  pressure favors association because the molecule combines two free
  particles.
- **Contract earned.** Positivity, limiting-reagent bounds, units, and the
  difference between a conditional mass-action population and a closed gas.
  The missing total-particle row is stated explicitly.
- **Causal next link.** The scalar symbols \(A\) and \(B\) now need to be tied
  to a real catalog record.

## Cell 3 — Decode CO code 608 without inventing a new notation

- **Remembered question.** How does the exact catalog say that CO spends
  carbon and oxygen?
- **New claim.** The integer molecular code is a compact base-100 component
  encoding, while component arrays—not decimal guessing—are authoritative for
  multiplicity and equation placement.
- **Minimal notation.** Use atomic numbers \(Z_{\rm C}=6\) and \(Z_{\rm O}=8\);
  retain \(\nu_{m a}\) only for repeated components.
- **Exact code function/source shown.** Load the exact atmosphere record
  through `read_molecular_equilibrium_catalog`; inspect
  `component_start_indices`, `component_equation_indices`, and
  `equation_species_codes` for code 608. A two-step `divmod` is shown only as
  an explanatory check against those arrays.
- **Expected output.** A three-line diagnostic:
  `608 -> [6, 8] -> [C equation, O equation]`, component count 2, and one
  boolean confirming the synthesis record has the same component semantics.
- **Immediate interpretation.** CO is one catalog population but enters two
  conservation rows. Code order is not a license to compare the two catalogs
  by row number.
- **Contract earned.** Real CO has replaced the anonymous \(AB\) example, and
  source component metadata owns the mapping.
- **Causal next link.** Before solving the coupled budgets, determine what the
  exact backends mean by \(K_m(T)\), especially at policy boundaries.

## Cell 4 — See the three H2 formation policies and their gates

- **Remembered question.** Is a formation constant one universal function of
  temperature?
- **New claim.** The atmosphere H2 table, the synthesis catalog polynomial,
  and the provisional named-H2 estimate are three exact policies with
  different activation edges; none may silently replace another.
- **Minimal notation.** Introduce \(K_m(T)\), component count \(C_m\), and the
  phrase “raw formation constant.” The prose derives the dimensionless
  exponent, rather than presenting coefficients without units:
  \[
  \ln K_m =
  \frac{a_0}{k_{\rm B}T\,[{\rm eV}]}-a_1+a_2T-a_3T^2+a_4T^3-a_5T^4+a_6T^5
  -\frac32(C_m-2q_m-1)\ln T ,
  \]
  with the atmosphere six-coefficient truncation stated alongside it. H2's
  table-based expression is read as translational phase space times its
  partition and dissociation factors. Do not introduce line-population
  normalization.
- **Exact code function/source shown.** Evaluate
  `payne_zero_atmosphere.molecular_equilibrium.hydrogen_molecule_equilibrium_constant`
  and `compute_equilibrium_constants_for_layer`, plus
  `payne_zero_synthesis.molecular_equilibrium.polynomial_formation_constants`.
  The provisional 4.478-eV code-101 policy is exposed under its physical name
  H2; the misleading local source variable `is_hminus` is identified as a
  naming wart, never taught as H-minus. The same cell names the two
  zero-leading-coefficient branches once: one-component records receive
  \(K=1\); multi-component atomic-ion records receive Saha-derived constants
  from the current backend's electron seed.
- **Expected output.** One professional, one-panel plot of
  \(\log_{10}\) raw H2 formation constants versus temperature, using the book
  palette and explicit gate markers. A small printed boundary table checks
  9000, 10000, 19900, and 20000 K with their next representable neighbors.
  The atmosphere H2 interpolation clamps below 100 K, uses the special table
  through its endpoint behavior, and is inactive only above 20000 K. The
  diagnostic explicitly records that the exact `T >= 19900` clamp/index rule
  selects the lower of the last two tabulated entries at zero interpolation
  fraction; a generic `np.interp` replacement is not accepted without parity.
  Atmosphere ordinary polynomials become exactly zero above 10000 K;
  synthesis zeros later become the finite log sentinel `-700`.
- **Immediate interpretation.** A smooth plotting curve cannot prove a branch
  edge. Raw \(K_m\), raw \(n_m\), and \(n_m/U_m\) are three different
  quantities; only the first is on this axis. Ordinary atmosphere polynomial
  exponent overflow is unguarded, whereas the controlled H2 branch is guarded;
  malformed overflow probes never enter a golden.
- **Contract earned.** Exact formation-policy ownership and branch gates are
  fixed once. Later cells refer back to this result rather than replotting it.
- **Causal next link.** A real \(K_{\rm CO}\) now predicts CO for a trial
  \((x_{\rm C},x_{\rm O})\); conservation must choose that pair.

## Cell 5 — Build the smallest coupled residual

- **Remembered question.** Why not solve the C and O ledgers one at a time?
- **New claim.** CO makes particle, carbon, and oxygen conservation one vector
  root. Repairing only one elemental ledger changes the CO population and
  reopens the other ledger.
- **Minimal notation.**
  \[
  \begin{aligned}
  R_0 &= x_{\rm C}+x_{\rm O}+n_{\rm CO}-P_{\rm gas}/(kT),\\
  R_{\rm C} &= x_{\rm C}-A_{\rm C}x_0+n_{\rm CO},\\
  R_{\rm O} &= x_{\rm O}-A_{\rm O}x_0+n_{\rm CO},\\
  n_{\rm CO} &= K_{\rm CO}x_{\rm C}x_{\rm O}.
  \end{aligned}
  \]
- **Exact code function/source shown.** A chapter-local `co_residual` mirrors
  the corresponding rows of atmosphere `_newton_matrix_kernel`; the visible
  function is at most 20 lines and names every term.
- **Expected output.** Print the initial three residuals, the state after a
  carbon-only repair, and the state after an oxygen-only repair. A final
  simultaneous root has all three normalized residuals below the declared
  threshold. The coupled-budget schematic sits directly above this cell.
- **Immediate interpretation.** The repeated \(n_{\rm CO}\) term is the
  coupling. Newton's method is motivated by the failed independent repairs,
  not introduced as an abstract optimization tool.
- **Contract earned.** The reader can read the production residual's physical
  rows before seeing its 23-dimensional Jacobian.
- **Causal next link.** A vector root needs the sensitivity of every ledger to
  every free density.

## Movement I close

Close in one paragraph: mass action predicts a molecule for a trial state;
conservation chooses the state; one shared molecule turns separate elemental
ledgers into a coupled problem. Do not summarize backend policies yet.

# Movement II — Keep one coupled root positive through depth

## Movement question

How does the atmosphere implementation turn the readable residual into a
stable, positive six-depth calculation without pretending that the depth rows
are independent?

## Movement claim

At one depth the atmosphere backend assembles an analytic Jacobian, solves an
unscaled linear system, and applies an order-sensitive density-space recovery
rule. Across depth it uses pressure-scaled continuation. Numba compiles the
local kernels, but neither `prange` nor a GPU is used for this ordered
molecular solve.

Place the second original schematic before Cell 8:

`assets/schematics/textbook/ch04-newton-positivity-v1.png`

It shows `residual → Jacobian → delta → density update`. The final arrow forks
into an atmosphere branch labeled “reflection/shared scale” and a synthesis
branch labeled “multiplicative floor.” It must not imply that either backend
uses logarithmic unknowns.

## Cell 6 — Differentiate one exact atmosphere residual

- **Remembered question.** What information does Newton need beyond the
  residual values?
- **New claim.** The Jacobian records how one density changes every budget;
  off-diagonal CO derivatives are the numerical signature of chemical
  coupling.
- **Minimal notation.**
  \(J_{ij}=\partial R_i/\partial x_j\) and
  \(J\boldsymbol{\delta}=\mathbf{R}\). Derive the three-by-three CO block
  before displaying any 23-row diagnostic.
- **Exact code function/source shown.** Use the exact atmosphere
  `_newton_matrix_kernel` through a narrow chapter runtime adapter. The reader
  sees the analytic CO sub-block and the call boundary, not a dump of the
  complete kernel.
- **Expected output.** A labeled three-by-three CO Jacobian, its independently
  finite-differenced counterpart, maximum absolute and scale-relative
  differences, and a compact full-23-row finite-difference summary.
- **Immediate interpretation.** The diagonal ones come from free-density
  ledgers; the cross terms come from
  \(\partial(Kx_{\rm C}x_{\rm O})/\partial x_a\). Agreement tests the
  derivative, not convergence.
- **Contract earned.** The analytic atmosphere residual/Jacobian pair has an
  independent numerical check.
- **Causal next link.** A valid Jacobian still has to be turned into an update,
  including a defined response to singularity.

## Cell 7 — Solve the atmosphere step and force its fallback

- **Remembered question.** What does the code do with
  \(J\boldsymbol{\delta}=\mathbf{R}\)?
- **New claim.** The atmosphere route first uses the unscaled NumPy
  `float64` solve and falls back to least squares only when the direct solve
  raises `LinAlgError`.
- **Minimal notation.** Reuse \(J\), \(\mathbf{R}\), and
  \(\boldsymbol{\delta}\); add the linear residual
  \(\|J\delta-R\|\), not a new physical residual.
- **Exact code function/source shown.** Execute the exact branch used inside
  `solve_molecular_equilibrium_layer`: `np.linalg.solve`, followed by
  `np.linalg.lstsq(..., rcond=None)` in the exception path. Use one physical
  nonsingular matrix and one controlled rank-deficient copy.
- **Expected output.** Two rows: branch taken, matrix rank, step norm, and
  linear residual norm. The forced fallback is a diagnostic only and is not
  substituted into a scientific golden.
- **Immediate interpretation.** Least squares makes a step available for a
  singular linearization; it does not prove that the nonlinear chemistry is
  physically closed.
- **Contract earned.** The exact unscaled LAPACK boundary and fallback are
  visible and separately labeled.
- **Causal next link.** Even a mathematically valid step can cross zero or
  oscillate, so the density update needs its own contract.

## Cell 8 — Trigger every atmosphere positivity branch

- **Remembered question.** If Newton proposes a negative density, does the
  solver change variables?
- **New claim.** The atmosphere solver remains in density space. It checks
  relative update against \(10^{-4}\), damps sign reversals by 0.69, reflects
  sufficiently large negative candidates with `abs`, and otherwise applies an
  order-dependent shared scale.
- **Minimal notation.** Use
  \(x_i^{\rm candidate}=x_i-\delta_i\) and the one-percent floor
  \(x_i/100\). No logarithmic variable is introduced.
- **Exact code function/source shown.** Feed controlled vectors into the exact
  atmosphere `_newton_update_kernel`. Indices are chosen to trigger, in order:
  ordinary accept, sign flip, absolute-value reflection, one-percent fallback,
  and the later `sqrt(scale)` consequence.
- **Expected output.** A five-row table with old density, raw delta, effective
  delta, branch label, new density, and shared scale seen by that index. All
  outputs are positive and the expected `0.69`, `100`, and `10` values appear.
- **Immediate interpretation.** Reflection is not clipping and the shared
  scale makes vector order part of parity. This policy must not be “cleaned
  up” into the synthesis floor.
- **Contract earned.** Every exact atmosphere convergence/damping/positivity
  branch has an executable witness.
- **Causal next link.** The update rule can now be trusted inside a real
  one-depth chemical solve.

## Cell 9 — Check physical directions and array meanings independently

- **Remembered question.** Does the exact one-depth result behave like the
  transparent association reaction?
- **New claim.** Independent temperature and pressure controls test chemistry
  without continuation history, while raw populations, saved physical
  equation densities, transformed equation columns, and partition-normalized
  populations must remain distinct.
- **Minimal notation.** Reuse \(n_m\), \(x_i\), and \(n_m/U_m\). Define the
  normalized residual
  \(\max_i |R_i|/\max(\hbox{row scale}_i,10^{-300})\).
  Derive one representative live-basis transformation,
  \[
  \widetilde x_a =
  \frac{x_a}{U_{a,0}\,1.8786\times10^{20}(m_a^{\rm amu}T)^{3/2}},
  \]
  and the corresponding electron transformation. This is the point at which
  the units change; it is not left as an unexplained array mutation.
- **Exact code function/source shown.** Allocate with
  `initialize_molecular_equilibrium_state`, run atmosphere
  `solve_molecular_equilibrium(..., population_mode=1)` as separate one-depth
  calls, and probe `populate_molecular_species` for the named molecules.
  Instrument iteration counts without changing the exact solve. The prose
  reads the dispatch once: codes at least 100 use a molecule-code lookup to
  `1e-3`; modes 1/11 select normalized populations and other modes select raw
  populations. For codes below 100, an absent catalog family falls back to
  atomic Saha, while a present family with a missing stage writes zero.
- **Expected output.** Two compact tables: three temperatures at fixed pressure
  and three pressures at fixed temperature. Each reports convergence before
  exhaustion, normalized residual, particle/element/charge conservation,
  raw H2/CO, one saved physical equation density, its transformed live value,
  and one normalized molecular value. Print exact atmosphere shapes
  `(D,170)` and `(D,23)`. Separate deterministic rows straddle the 10000 K and
  20000 K gates. Modes 2/12 are not used here.
- **Immediate interpretation.** Cool/high-pressure trends are conditional on
  the exact \(K_m\) policies already seen in Cell 4. The transformed equation
  column is no longer cm\(^{-3}\); the saved copy and live column 0 remain
  physical. Constants used during Newton are frozen from the seed state, raw
  populations recompute constants after the solved \(n_e\), and
  `charge_square_density` remains the pre-solve seed. These are limitations,
  not hidden iterations.
- **Contract earned.** One-depth scientific closure, exact gate behavior, and
  the atmosphere array lifecycle are all independently checked. Reaching the
  iteration cap is an explicit chapter failure even though the production
  function returns its last vector.
- **Causal next link.** Real atmospheres are ordered columns, so the next
  question is how one converged row seeds another.

## Cell 10 — Follow the root through six ordered depths

- **Remembered question.** Chapter 3 used `prange` over independent atom-only
  depths. Are molecular depths equally independent?
- **New claim.** Production molecular chemistry is a sequential,
  pressure-scaled continuation:
  \(\mathbf{x}^{(0)}_d=\mathbf{x}^{(*)}_{d-1}P_d/P_{d-1}\).
  Deliberate one-depth restarts are diagnostics, not the production route.
- **Minimal notation.** Add only the depth index \(d\) and the pressure ratio.
- **Exact code function/source shown.** Run atmosphere
  `solve_molecular_equilibrium` on the fixed six-depth track, then run six
  deliberate one-depth restarts for comparison. Identify the four exact
  `@njit(cache=True, nogil=True)` molecular kernels: equilibrium constants,
  residual/Jacobian, update, and energy. `np.linalg.solve/lstsq` remains
  outside them.
- **Expected output.** One one-panel plot of Newton iterations versus depth
  for continuation and restart, with the scientific convergence threshold
  printed separately. Report warmed CPU time only as a local diagnostic.
- **Immediate interpretation.** `njit` removes Python overhead from repeated
  local arithmetic. `prange` would be incorrect here because depth \(d\)
  consumes depth \(d-1\); none of these molecular kernels has
  `parallel=True`, and there is no atmosphere molecular GPU route. Batched
  Saha table evaluation elsewhere does not turn the Newton solve into a
  depth-batched solve. At a new atmosphere call the initial nuclei density is
  \(P/(2kT)\), except below 4000 K where it is \(P/(kT)\); the initial electron
  scale is one tenth of that. A new `MolecularEquilibriumState` is allocated
  for every outer atmosphere iteration, so continuation crosses depths within
  one call, not outer iterations.
- **Contract earned.** Depth order, pressure continuation, Numba's useful
  boundary, the absence of molecular `prange`, and the absence of an
  atmosphere GPU claim are explicit.
- **Causal next link.** Continuation normally starts from the preceding depth,
  but thermal-energy perturbations deliberately restore a saved physical
  state.

## Cell 11 — Preserve physical seeds and earn molecular internal energy

- **Remembered question.** What must survive after the live equation array is
  transformed for line-population normalization?
- **New claim.** A separate saved `(D,23)` physical-density array owns warm
  starts. In specific-energy mode, a nonzero saved row overrides the
  pressure-scaled seed, is not overwritten, and causes an early return after
  the full molecular energy is evaluated.
- **Minimal notation.** Reuse the Chapter 3 meaning of specific internal
  energy. Introduce only the molecular partition-response factor
  \[
  \frac{Q(1.001T)-Q(0.999T)}
       {\max[Q(1.001T)+Q(0.999T),10^{-30}]}\,1000 .
  \]
  The chapter then reads
  \(u=[1.5P_{\rm gas}+\sum_m n_m(E_{{\rm bind},m}
  +kT\,\partial\ln Q_m/\partial\ln T)]/\rho\), with the exact H2
  `36118.11 cm^-1` dissociation term as the concrete example.
- **Exact code function/source shown.** Use
  `save_molecular_equation_density`,
  `restore_molecular_equation_density`,
  `set_molecular_specific_internal_energy_mode`, and
  `compute_molecular_specific_internal_energy`. Demonstrate the exact early
  returns in `solve_molecular_equilibrium` for modes 2/12 and energy mode.
- **Expected output.** One depth checkpoint prints physical saved density,
  transformed live density, restored seed, unchanged saved-row hash, raw H2
  and CO populations, translation-only \(1.5P/\rho\), full molecular
  `specific_internal_energy`, and the energy-mode seed source. A mode table
  confirms that normalized arrays and normal translation assignment are
  skipped on the exact early-return paths.
- **Immediate interpretation.** Normal population filling stores
  translation-only energy; the molecular dissociation/partition contribution
  belongs to the dedicated energy mode. The public runner defaults
  `molecular_convection_thermal_tracks_perturbation=True`, while lower-level
  helpers default it to `False`; a scientific example must use
  `run_atmosphere_model` or pass the flag explicitly. Convection physics
  remains deferred. The orchestration's `finally` path restores the saved
  runtime, molecular arrays, and temperature cache; if no original energy was
  present it uses Chapter 3's atomic fallback rather than inventing a second
  molecular calculation.
- **Contract earned.** Physical warm-start ownership, transformed-array
  ownership, early-return behavior, and molecular energy are taught exactly
  once.
- **Causal next link.** The atmosphere root is now physically and numerically
  intelligible. The synthesis backend solves the same kind of root with a
  different derivative and positivity implementation.

## Movement II close

Close with a short contrast: the atmosphere backend is CPU NumPy `float64`;
Numba compiles local kernels; LAPACK solves the linear system; ordered
continuation prevents molecular depth parallelism. Do not preview catalog
counts or public lane numbers here.

# Movement III — Cross the atmosphere/synthesis boundary honestly

## Movement question

How can the device-capable synthesis implementation preserve the coupled
chemistry while making different numerical and public-state choices, and
which route is allowed to own the electron density?

## Movement claim

Synthesis still solves for ordinary densities and still walks depths in order.
It evaluates only molecular products in logarithms, obtains a one-depth
Jacobian with `jacrev`, uses a column-scaled linear solve, and calls `vmap`
only after every depth loop has ended. Full, fixed-public-\(n_e\), live
atmosphere, diagnostic, and release routes make different state claims.

Place the third original schematic before Cell 12:

`assets/schematics/textbook/ch04-ordered-backends-v1.png`

Two horizontal depth chains share the same outer-to-inner direction. The
atmosphere chain labels Numba-local kernels and CPU LAPACK. The synthesis chain
places `jacrev` inside one depth/iteration box and places `vmap` only after the
final depth. No all-depth Newton arrow is allowed.

## Cell 12 — Execute one exact synthesis Newton step

- **Remembered question.** Which parts of the atmosphere Newton logic are
  physical, and which are backend policy?
- **New claim.** Synthesis uses the same density-space root but computes
  products with `_safe_log`, differentiates one depth with `jacrev`, solves a
  density-column-scaled system, and floors an invalid candidate at
  `current_density/100`. Only multi-component records enter the nonlinear
  residual; the final population evaluation includes one-component atoms and
  ions too.
- **Minimal notation.**
  \[
  \ln n_m=\ln K_m+\sum_i\nu_{mi}\ln x_i-q_m\ln x_e.
  \]
  Now introduce \(x_e\), inverse-electron power \(q_m\), and the negative-ion
  correction. Explain catalog code 100 as the ordinary electron component and
  code 101 in `species_to_equation_index` as the one-past-active
  inverse-electron sentinel; this sentinel meaning is distinct from molecule
  code 101 for H2.
- **Exact code function/source shown.** Call exact synthesis `_residual` and
  `_newton_step` through a narrow adapter. Compute `jacrev(_residual,
  argnums=0)` for one controlled depth and compare with independent finite
  differences. Trigger the synthesis sign-flip `0.69` and multiplicative
  positive floor. Show the exact chain-start seed and the
  `max(elemental_abundance, 1e-20)` network floor; use an explicit
  `chain_length` restart once, separately from the production default chain.
- **Expected output.** Print density-vector units/dtype, analytic-versus-finite
  Jacobian error, raw and scaled linear residuals, maximum fractional update,
  and a branch table. Print the cold/hot chain-start values
  (\(x_0=P/kT\) below 4000 K, otherwise \(P/(2kT)\), with base \(x_0/10\))
  and confirm the per-network abundance floor without changing the public
  abundance vector. A sentinel diagnostic covers all 14 atmosphere and
  synthesis negative-ion component-order invariants. A controlled overflow
  probe reports the pre-replacement nonfinite product, the residual's
  replacement behavior, and the final finiteness gate separately.
- **Immediate interpretation.** Logarithms stabilize the product, not the
  unknown. Column scaling changes conditioning, not the nonlinear equations.
  Unlike atmosphere recovery, synthesis has no absolute-value reflection and
  no cross-index shared scale. Nonfinite residual products are replaced by
  zero, but final `_molecular_densities` has no equivalent guard; this
  asymmetry is surfaced as a limitation. Direct
  `solve_molecular_equilibrium` defaults to float32; public APIs default to
  float64 away from MPS. A direct MPS float64 request silently substitutes
  float32, whereas the public runtime rejects explicit MPS float64. Scientific
  direct calls always state dtype and device. The direct molecular defaults
  are `max_iter=200`, `tol=1e-3`, and one full-depth chain; the public EOS
  route uses `max_iter=200` and `tol=1e-4`.
- **Contract earned.** Density-space Newton, per-iteration `jacrev`, exact
  scaling, charge encoding, and synthesis positivity are independently
  verified.
- **Causal next link.** One step is not a public state. The full route must
  reveal every molecular solve that contributes to its published result.

## Cell 13 — Run the full synthesis route and retain both solves

- **Remembered question.** When synthesis is asked to close the whole
  molecule-enabled gas, which solve owns its published \(n_e\)?
- **New claim.** `solve_population_state(..., molecules=True)` makes two
  ordered molecular solves: the first occurs inside electron-density closure
  and supplies published \(n_e\); the second assembles molecular arrays and
  the molecule-backed population state.
- **Minimal notation.** Distinguish
  \(n_e^{\rm published}\), \(n_e^{(1)}\), and internal
  \(n_e^{(2)}\). Do not create a generic “true electron density.”
- **Exact code function/source shown.** Use exact
  `payne_zero_synthesis.equation_of_state.solve_population_state` with a
  read-only call-capture wrapper around both exact
  `solve_molecular_equilibrium` invocations. Also expose
  `molecular_seed_electron_density` and
  `molecular_ion_formation_constants_from_seed`: the first molecular seed is
  \(P/(20k_{\rm B,ref}T)\), later seeds use the preceding upstream
  \(n_e P_d/P_{d-1}\), and atomic-ion constants use
  \((f_{Z,q}/f_{Z,0})n_{e,\rm seed}^{q}\). Every scientific direct call states
  device and dtype explicitly.
- **Expected output.** A two-row call trace shows order, seed, requested and
  effective dtype, `tol`, `max_iter`, `chain_length`, iteration vector, and
  returned electron column. Then print
  public residuals; selected H2/CO values; public `(D,6,139)` cubes; padded
  `molecular_populations (D,200)` and
  `molecular_equation_densities (D,30)`; and exact zeros in the last 10 and 7
  columns. Check one molecule-table-absent atomic family, whose atomic
  fallback is rescaled by molecular/atomic nuclei density, and one
  present-family missing stage, which remains zero. Only after the local
  result exists, compare every field with the CPU-float64 golden, then report
  separately measured CPU-float32 and available CUDA/MPS profiles. An
  unavailable accelerator receives no claim.
- **Immediate interpretation.** The two solves are an implementation seam,
  not duplicate evidence of closure. They can request different dtypes for an
  explicitly float32 off-MPS table bundle: the first requests reference
  float64 and assembly uses `tables.dtype`. Iteration exhaustion returns the
  last vector but fails the textbook acceptance gate.
- **Contract earned.** The full route's electron ownership, two-solve
  lifecycle, active/padded shapes, finiteness, residual, dtype, and parity
  claims are all preserved.
- **Causal next link.** A structured atmosphere may already own \(n_e\);
  synthesis then needs a route that preserves the public input without
  pretending its internal molecular equation used the same value.

## Cell 14 — Run the fixed-public-\(n_e\) route

- **Remembered question.** Does “fixed electron density” mean the molecular
  root contains no electron unknown?
- **New claim.** No. `solve_population_state_at_electron_density` preserves
  the supplied **public** electron column exactly, while its internal
  molecular solve has its own electron equation and discards that internal
  electron on return.
- **Minimal notation.** Distinguish \(n_e^{\rm input}\),
  \(n_e^{\rm public}\), and \(n_e^{\rm internal}\).
- **Exact code function/source shown.** Run exact
  `solve_population_state_at_electron_density` twice: once with supplied
  `mass_density` and once without it. Capture the one internal exact
  molecular solve without changing results.
- **Expected output.** A compact two-branch table proves bitwise
  \(n_e^{\rm public}=n_e^{\rm input}\), reports the difference from
  \(n_e^{\rm internal}\), identifies supplied versus composition-derived mass
  density, checks all shapes/padding/finiteness, rejects exhaustion, and then
  compares with the fixed-route golden.
- **Immediate interpretation.** Public preservation is an interface contract,
  not an internal charge-closure claim. The solver's nuclei density remains
  public in both mass branches; only mass-density ownership changes.
- **Contract earned.** Fixed-public-\(n_e\) semantics and both mass branches
  can no longer be confused with full closure.
- **Causal next link.** The two synthesis calls still do not describe the live
  atmosphere handoff, disabled-pressure route, or release builder. Put all
  routes in one claim matrix.

## Cell 15 — Audit route ownership instead of blending it

- **Remembered question.** Which entry point may move \(n_e\), fill
  populations, reuse molecular arrays, or choose an H2 fallback?
- **New claim.** Route names are scientific contracts. Similar-shaped outputs
  do not grant the same closure or preservation claim.
- **Minimal notation.** No new algebra. Use only input, internal, and public
  ownership labels already earned.
- **Exact code function/source shown.** Instrument exact atmosphere
  `prepare_population_state` and
  `prepare_structured_handoff_population_state`, atmosphere bridge
  `_molecular_hydrogen_population`, and synthesis
  `build_structured_atmosphere_from_columns`.
- **Expected output.** One route-claim matrix with these exact rows:
  1. molecule-enabled atmosphere population phase: seed overwritten, one
     ordered solve;
  2. atmosphere pressure iteration disabled: fresh seed retained, molecular
     storage may exist, atomic/molecular populations remain empty, Doppler
     support still runs;
  3. molecule-enabled atmosphere structured handoff: input \(n_e\) may move
     during one coupled solve, then packed populations are refilled at the new
     density;
  4. synthesis full route: two solves, first electron published;
  5. synthesis fixed route: public input preserved, internal electron
     discarded;
  6. public molecular-lines builder: exact edge grid loaded, one fixed solve,
     returned molecular arrays reused, fallback re-solve not entered.

  The same cell prints two H2 bridge probes: a mixed positive/zero molecular
  vector is selected globally, including its zeros; only an all-zero vector
  selects the analytic provisional H2 estimate. It also source-probes the
  live/debug `(200,)` code versus `(D,170)` population mismatch and labels that
  path incompatible. Release output is independently rebuilt by synthesis
  with its 190-row catalog. A compact call-count line also proves the
  atmosphere zero-code priming call performs the one solve for a temperature
  index and the following 230-job population schedule consumes the cached
  state without resolving. One final atmosphere support probe preserves exact
  structural `+inf` Doppler widths at zero-based packed slots 919 and 927,
  proves their major-isotope masses plus actual and normalized populations are
  exactly zero, and requires every other Doppler slot to be finite. These
  structural infinities are labeled rather than sanitized.
- **Immediate interpretation.** “Fixed,” “handoff,” “diagnostic,” and
  “release” are not synonyms. The pressure-disabled route is not a hidden
  fixed-density fill. The H2 fallback is global rather than depth-wise.
- **Contract earned.** Full/public/fixed ownership, empty-fill behavior,
  edge-grid dependency, exact solve count/reuse, live incompatibility, and
  release boundary are explicit.
- **Causal next link.** Route ownership is clear; the remaining question is
  how the 170- and 190-record catalogs relate chemically.

Place the fourth original schematic immediately after Cell 15:

`assets/schematics/textbook/ch04-catalog-to-public-lane-v1.png`

It shows a 170-record atmosphere set inside the shared region, a 190-record
synthesis set with 20 additional records, and a separate arrow from 54
line-list species mappings into selected cells of the normalized stage-5
sheet. That fixed-stage sheet spans depth by species, with 51 selected columns
inside 0–98 and three at 129–131; a separate card says the actual ion-stage
cube is unchanged. It must not draw catalog row number as a public species
index or a constant-depth sheet as stage index 5.

## Cell 16 — Align catalogs by code and semantics

- **Remembered question.** Are atmosphere row 45 and synthesis row 45
  necessarily the same chemical record?
- **New claim.** Catalogs compare by integerized molecule code plus component,
  coefficient, and sentinel semantics—not by row.
- **Minimal notation.** Use the already defined catalog record \(m\); no new
  physics symbol.
- **Exact code function/source shown.** Read with
  `read_molecular_equilibrium_catalog` and `read_molecule_table`; construct
  immutable code-key dictionaries and compare active records. Validate
  component bounds, atmosphere coefficient row 6, active padding, and the
  reverse-lookup `-1`/electron/inverse-electron exceptions separately.
- **Expected output.** Print `shared=170`, `atmosphere-only=0`,
  `synthesis-only=20`, and zero shared semantic mismatches. Print the exact
  synthesis-only codes:
  `111`, `10811`, `10812`, `10820`, `60606`, `60608`, `60614`, `60816`,
  `61414`, `61616`, `70708`, `70808`, `80814`, `80816`, `1010106`,
  `1010107`, `1010606`, `6060707`, `101010106`, `101010114`.
  A separate padding line names every truly zero-padded region and confirms
  that the atmosphere reverse lookup instead uses `-1`, electron-row, and
  one-past-active sentinel values.
- **Immediate interpretation.** The synthesis catalog is a strict
  code-semantic extension for this pinned pair, not a row-appended format
  promise for arbitrary future catalogs.
- **Contract earned.** Shared chemistry, synthesis-only chemistry, padding,
  component bounds, and sentinel semantics are independently verified.
- **Causal next link.** Catalog records are internal chemistry. The final cell
  maps normalized molecular populations into the exact public lane consumed
  next.

## Cell 17 — Map CO and verify every public molecular lane

- **Remembered question.** Which molecular quantity crosses into the
  synthesis-ready public population cube?
- **New claim.** Molecular line support uses \(N_{\rm mol}/U_{\rm mol}\), not
  raw \(N_{\rm mol}\). The release builder leaves the actual ion-stage cube
  atomic and repurposes only 54 selected cells in stage index 5 of the
  partition-normalized cube. This is an overloaded synthetic interface lane,
  not a separate molecular block or a sixth molecular ion stage.
- **Minimal notation.** Reuse \(N_m/U_m\). State once that index 5 is an
  overloaded synthetic storage lane in the normalized cube, while the actual
  cube retains atomic sixth-stage values at the same coordinates. Derive the
  line basis once:
  ordinary elemental and electron equation densities are transformed by
  their exact translational/partition denominators, then a polynomial molecule
  rebuilds \(N_m/U_m\) from that product and its molecular-mass translational
  factor. A zero-polynomial atomic-ion record instead uses its raw population
  divided by the relevant partition.
- **Exact code function/source shown.** Use
  `molecular_line_populations`,
  `molecular_line_populations_by_species_code`, and the exact pipeline mapping
  assembled by `build_structured_atmosphere_from_columns`. Recompute
  molecular-element partitions with `apply_ground_partition=False`, exactly
  as the pipeline does.
- **Expected output.** Trace line-list species 276 to equilibrium code 608 and
  destination `[depth, 5, 45]`. Prove its normalized-cube value equals the
  independently calculated normalized CO population and differs from raw CO
  where expected. Verify all 54 unique
  species/equilibrium/public-column mappings and print their exact 51/3 split:
  51 destination columns lie in 0–98 and only columns 129–131 lie in the
  tail that is unused by the atom-only state. Compare the complete pre/post
  cubes:
  `ion_stage_populations` is unchanged everywhere; the molecular delta in
  `partition_normalized_populations` is zero in stage indices 0–4 and at every
  unowned stage-5 cell; each owned cell equals the no-ground result. Include
  the ground-partition-off normalization checkpoint.
  Confirm the hard-coded `(99,)`
  `_ATOMIC_MASSES_FOR_MOLECULES` raw-byte hash and one normalization value;
  `atomic_masses.npz` is not allowed to substitute for that parity authority.
  Verify the atmosphere molecule-enabled schedule has 230 jobs: 198 inherited
  atomic jobs plus modes 1 and 11 for each of the exact 16 selected molecular
  codes, with both modes reading the same normalized source and landing in
  their pinned one-based packed slots:
  `101→841`, `106→846`, `107→847`, `108→848`, `112→851`, `114→853`,
  `120→858`, `124→862`, `126→864`, `606→868`, `607→869`, `608→870`,
  `814→889`, `822→895`, `823→896`, and `10108→940`. Finally print the
  grouped Chapter 5 handoff fields and their axes. The exact inventory is:

  - state: `temperature`, `gas_pressure`, `electron_density`, `mass_density`,
    `column_mass`;
  - populations/support:
    `partition_normalized_populations`, `ion_stage_populations`,
    `fractional_doppler_widths`, `hydrogen_neutral_population`,
    `helium_neutral_population`, `helium_singly_ionized_population`,
    `molecular_hydrogen_population`,
    `hydrogen_partition_normalized_ion_stage_populations`,
    `carbon_partition_normalized_ion_stage_populations`,
    `magnesium_neutral_partition_normalized_population`,
    `aluminum_neutral_partition_normalized_population`,
    `silicon_neutral_partition_normalized_population`,
    `iron_neutral_partition_normalized_population`;
  - thermal/mixture: `hc_over_kt`, `microturbulence`,
    `elemental_abundances`;
  - edge grid: `signed_continuum_edge_frequency_hz`,
    `continuum_edge_wavelength_nm`,
    `continuum_edge_midpoint_wavelength_nm`,
    `continuum_edge_interval_width_squared_over_two_nm2`.
- **Immediate interpretation.** Chemistry catalog code, line-list species
  code, public species column, and synthetic-lane index are four different
  identifiers. At an owned stage-5 coordinate the actual and normalized cubes
  no longer form an \(n\) and \(n/U\) pair for one species: the actual value
  remains atomic while the normalized value is molecular line support.
  Atmosphere's 16 duplicated packed molecular jobs are a limited legacy
  opacity support path; modes 1 and 11 select the same normalized source. The
  synthesis public mapping verifies 54 line-list mappings and is the release
  path.
- **Contract earned.** The complete molecular lifecycle reaches an exact,
  normalized, synthesis-ready representation with no raw/normalized or
  ion/synthetic-axis ambiguity.
- **Causal next link.** Chapter 5 can now ask how these particles absorb and
  scatter continuum photons.

## Movement III close and chapter summary

The summary introduces no new object, function, number, or limitation. It
returns to the opening CO problem and states only what the chapter has earned:

- one molecule couples several elemental ledgers into one density-space root;
- mass action supplies conditional populations, while simultaneous particle,
  element, and charge conservation choose the state;
- both exact backends preserve ordered pressure-scaled depth continuation;
- only synthesis molecular products are evaluated in logarithms; neither
  backend has logarithmic Newton unknowns;
- atmosphere and synthesis use intentionally different Jacobian, scaling, and
  positivity policies;
- atmosphere, full synthesis, fixed-public-\(n_e\), diagnostic, and release
  routes own different electron-density and population claims;
- the 170- and 190-record catalogs and the three H2 policies are exact,
  distinct inputs;
- raw molecular densities, physical equation densities, transformed equation
  columns, and partition-normalized line populations are not interchangeable;
- the public synthetic lane now carries verified \(N_{\rm mol}/U_{\rm mol}\)
  values for Chapter 5.

End with:

> **Next: let the particles interact with light.**  
> [Chapter 5: Continuum Absorption and Scattering](/reader.html?ch=5) starts
> from the closed, synthesis-ready state built here and asks which continuum
> processes remove or redirect photons. It will not reopen the molecular
> equilibrium solve.

# Visual and plotting contract

## Original schematics

All four schematics are newly generated from owned Python prompt/build scripts.
The website is an aesthetic reference only. Each asset needs its prompt,
generator provenance, SHA-256, alt text, caption, and scientific review record.

1. **Coupled budgets.** One CO token spends one C and one O token.
2. **Newton and positivity.** One residual/Jacobian path, then distinct
   atmosphere and synthesis update branches.
3. **Ordered backends.** Sequential depths in both backends; `jacrev` inside a
   synthesis depth, `vmap` after all depth loops.
4. **Catalog-to-lane boundary.** Shared 170, synthesis 190, 54 line mappings,
   one public synthetic lane.

The palette, typography, line weights, whitespace, and caption tone follow the
Payne Zero website schematics, but no website image is copied.

## Quantitative figures

There are exactly two required quantitative figures:

1. Cell 4: one panel, one claim—formation policies switch at different
   temperature gates.
2. Cell 10: one panel, one claim—continuation and deliberate restart have
   different iteration histories.

Both use `book.plot_style.single_panel`, `PAPER_COLORS`, restrained grids,
legible physical units, colorblind-distinguishable line styles, and print-safe
contrast. Neither has an inset, secondary axis, decorative background, or
multi-panel compression. Deterministic boundary tables, not plotted pixels,
own the exact gate assertions.

# Redundancy and forward-reference budget

## Allowed backward references

| Prior material | Chapter 4 allowance | Forbidden repetition |
| --- | --- | --- |
| Chapter 2 manifests | one compact asset preflight in Cell 1 | re-teaching SHA-256 or all data roles |
| Chapter 2 `njit`/`prange` | one contrast in Cell 10 | generic compilation tutorial or another parallel timing lesson |
| Chapter 3 particle and charge budgets | one opening reminder, then molecular extension | repeating the atom-only fixed-point derivation |
| Chapter 3 Saha and partitions | exact calls and one reminder of \(n/U\) | Saha derivation, partition branch tour, or pressure-lowering lesson |
| Chapter 3 layouts | only molecular additions and lane mapping | repeating the packed atomic sentinel demonstration |
| Chapter 3 atomic energy | reuse its meaning and add molecular terms once | re-deriving cumulative atomic ionization energy |
| Chapter 3 full/fixed routes | molecular ownership delta only | repeating atom-only route examples |

## Within-chapter repetition budget

- Mass action is derived in Cell 2, attached to CO in Cell 3, and thereafter
  referenced rather than re-derived.
- Formation policies are explained in Cell 4. Cells 9 and 17 may name the
  relevant policy but may not repeat its polynomial.
- The physical residual is derived in Cell 5, differentiated in Cell 6, and
  extended with charge encoding in Cell 12. These are successive additions,
  not three introductions.
- Atmosphere positivity is executed in Cell 8; synthesis positivity is
  contrasted once in Cell 12.
- Full and fixed synthesis routes are calculated in Cells 13 and 14. Cell 15
  only compares already-earned claims and adds the other exact routes.
- Raw, physical-equation, transformed-equation, and normalized populations are
  contrasted in Cell 9 and then used, without redefinition, in Cell 17.

## Forward-reference budget

- Chapter 5 is named only in the opening scope sentence, Cell 17's handoff, and
  the final link.
- Chapter 8 molecular line opacity is not named in the main narrative; the
  phrase “later line calculation” is sufficient.
- Chapter 12 convection appears only in Cell 11's ownership note explaining
  why energy perturbation seeds and the thermal-tracking flag exist. No
  convection equation is introduced.
- No later chapter is used to justify a result that Chapter 4 can test now.

# Acceptance checklist

## Pedagogical flow

- [ ] Exactly 17 visible scientific code cells appear in the stated order.
- [ ] Every cell declares reads, writes, units, axes, dtype/device, expected
      output, interpretation, earned contract, and causal next link.
- [ ] Visible code is at most 35 lines and 92 characters per line; helpers
      hide plumbing, never the scientific operation being taught.
- [ ] Every result is interpreted before the next abstraction appears.
- [ ] There are no exercises, source dumps, source tours, unexplained large
      code blocks, or “inspect this file” prose.
- [ ] The opening is concrete and the summary introduces nothing new.
- [ ] The final link points to Chapter 5 and says exactly what state it inherits.

## Physics and notation

- [ ] The first physical example is positive \(A+B\rightleftharpoons AB\), and
      its temperature claim is conditional on the declared \(K(T)\).
- [ ] CO 608 is decoded through catalog components as C plus O.
- [ ] Particle, element, and charge residual rows match the exact contracts.
- [ ] All Newton unknowns are ordinary cm\(^{-3}\) densities.
- [ ] Synthesis logarithms are confined to molecular product evaluation.
- [ ] Raw \(K_m\), raw \(n_m\), physical equation densities, transformed
      equation columns, and \(n_m/U_m\) are never conflated.
- [ ] The chapter wrapper rejects invalid inputs without renormalizing the
      valid Chapter 3 abundance vector.
- [ ] The wrapper validates positive finite thermodynamic columns and declared
      outer-to-inner order before either low-level continuation begins.

## Atmosphere implementation

- [ ] Analytic Jacobian and independent finite differences agree.
- [ ] Direct `np.linalg.solve` and forced `lstsq` fallback both execute.
- [ ] The exact \(10^{-4}\), 0.69, `abs`, one-percent, shared-100, and
      `sqrt(scale)` branches have visible witnesses.
- [ ] Independent temperature/pressure controls are separate from continuation.
- [ ] Continuation uses the previous converged row times the pressure ratio.
- [ ] Numba is described accurately; no molecular `prange` or GPU claim appears.
- [ ] Frozen Saha-derived constants, post-solve raw recomputation, and
      pre-solve `charge_square_density` are explicit.
- [ ] Normal, modes 2/12, and energy-mode array/return lifecycles are tested.
- [ ] Saved physical equation densities survive normalization and energy mode.
- [ ] Molecular energy and the public-true/lower-level-false thermal-tracking
      default asymmetry are tested exactly once.
- [ ] Exhaustion, nonfinite output, or failed conservation rejects the chapter
      result even where production returns a last vector.

## Synthesis implementation

- [ ] `jacrev` is called for one depth on every iteration; `vmap` appears only
      after all depth loops terminate.
- [ ] Column scaling, 0.69 damping, and the synthesis-only multiplicative floor
      execute visibly.
- [ ] Electron and inverse-electron sentinel meanings and all 14 negative-ion
      ordering invariants are pinned.
- [ ] Direct calls specify dtype/device; direct omitted-dtype float32 and MPS
      substitution are diagnosed, while public MPS float64 rejection remains
      distinct.
- [ ] The full route retains evidence from both molecular solves, including
      requested/effective dtype and iteration counts.
- [ ] The fixed route proves public input-\(n_e\) identity while making no
      internal charge-closure claim.
- [ ] Both fixed-route mass-density branches are exercised.
- [ ] Padded `(D,200)` and `(D,30)` arrays have exact last-10/last-7 zeros.
- [ ] Nonfinite residual replacement and unguarded final molecular evaluation
      are identified as separate behaviors.

## Catalogs, routes, and handoff

- [ ] All new assets and inherited dependencies match manifest identities.
- [ ] Atmosphere `170/23/481` and synthesis `190/23/548` active extents pass.
- [ ] The three H2 policies and all exact temperature boundaries pass.
- [ ] Catalogs align by code and semantics: 170 shared and the exact 20
      synthesis-only codes.
- [ ] True zero padding is distinguished from the atmosphere reverse lookup's
      `-1`, electron-row, and one-past-active sentinels.
- [ ] Molecule-enabled atmosphere handoff retains input and output \(n_e\) and
      makes no preservation claim.
- [ ] Pressure-iteration-disabled atmosphere state remains an empty population
      phase, not a fixed fill.
- [ ] Mixed/all-zero H2 vectors follow the exact global bridge selection.
- [ ] The live `(200,)`/`(D,170)` bridge incompatibility is source-probed and
      never used as the release route.
- [ ] The public builder loads the exact edge grid, performs one fixed solve,
      reuses returned arrays, and avoids the fallback re-solve.
- [ ] Atmosphere packed Doppler slots 919 and 927 preserve their exact
      structural `+inf` values with zero mass and zero actual/normalized
      populations; every other Doppler slot is finite.
- [ ] The hard-coded molecular-mass authority and ground-partition-off line
      normalization are independently pinned.
- [ ] CO lands at `[depth,5,45]`; all 54 mappings pass with the exact 51/3
      column split; the actual cube is unchanged and the normalized-cube delta
      is confined to the 54 owned stage-5 cells.
- [ ] Reader results exist before any of the five identity-bound comparison
      goldens are opened; key sets, shapes, dtype, axes, finiteness, physical
      residuals, backend differences, and zero leakage are reported separately.

## Visual and whole-chapter audit

- [ ] All four schematics are original, provenance-bound, scientifically
      audited, and readable at notebook width.
- [ ] Both quantitative plots are professional, one-panel, and one-claim.
- [ ] Exact branch checks are printed; no assertion depends on visual pixels.
- [ ] A prose pass removes duplicated definitions and premature exact names.
- [ ] A code pass removes hidden scientific work and oversized cells.
- [ ] A backward-reference pass enforces the table above.
- [ ] A forward-reference pass enforces the Chapter 5/8/12 budget above.
- [ ] A final zoom-out pass can trace every arrow in the global causal spine to
      at least one executed cell and every executed cell to exactly one arrow.
- [ ] The reviewed narrow Chapter 4 overrides, and no whole mixed-ownership
      modules, appear in `audit/paynezero_symbol_coverage.json`.
