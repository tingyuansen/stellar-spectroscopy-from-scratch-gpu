# Chapter 4 sequence-to-oracle reconciliation

Status: bounded audit of the 17-cell pedagogical sequence  
Sequence audited: `design/chapter04_pedagogical_sequence.md`  
Exact authority: `design/chapter04_exact_source_contract.md`  
Executable authorities:

- `scripts/chapter04_atmosphere_oracle_worker.py`;
- `tests/test_chapter04_atmosphere_oracle_worker.py`;
- `scripts/chapter04_synthesis_oracle_worker.py`;
- `tests/test_chapter04_synthesis_oracle_worker.py`;
- `tests/test_chapter04_source_data.py`;
- `book/chapter04_runtime.py` and `tests/test_chapter04_runtime.py`;
- `book/chapter04_teaching.py` and `tests/test_chapter04_teaching.py`.

This audit does not change the sequence, a worker, a test, a chapter, or either
external checkout.

## Executive verdict

The 17-cell causal order is strong and should remain:

```text
one shared molecule
    -> conditional mass action
    -> coupled conservation
    -> Jacobian and positive Newton update
    -> ordered atmosphere continuation
    -> synthesis's different numerical contract
    -> route ownership
    -> catalog alignment
    -> normalized public molecular lane
```

No cell is causally orphaned, and no new chapter is needed. The sequence also
passes the user's no-exercise requirement: useful checks are part of the main
argument rather than a detached exercise set.

The exact route evidence is substantially stronger than it was when the
sequence was written. In particular, the workers now support:

- the atmosphere full, structured-handoff, pressure-disabled, modes 2/12,
  molecular-energy, live-bridge, H2-boundary, and structural-Doppler routes;
- the synthesis full two-solve, fixed-public-electron, public one-solve/reuse,
  54-lane mapping, no-ground-partition, edge-grid, 10000 K, and provisional
  9000 K routes.

The chapter is not yet implementation-ready as written, however. Cells 6-8,
10, and especially 12 promise exact, visible kernel-level diagnostics that the
current reader runtime and oracle outputs do not yet provide. Cells 13, 14,
and 17 still promise comparison against unpublished goldens. Cells 9, 11, 12,
15, and 17 contain too many secondary implementation seams for one bite-sized
visible cell unless their prose is deliberately decompressed around a single
central calculation.

The correct response is not to add chapters or discard material. Keep 17
scientific cells, give each cell one numerical action, and move secondary
contract facts into short, evidence-bound prose or compact post-result tables.

## Verification snapshot

Under the repository's declared `PYTHONPATH=src:.` and dependency-bearing
Python runtime, the five focused Chapter 4 test modules passed:

```text
48 passed
```

Fresh in-memory pinned runs also completed for all four synthesis worker
routes:

| Route | Arrays | In-memory result digest |
| --- | ---: | --- |
| full | 132 | `b3ad13013a25157c40c97bf1d42ef092a8b201f6d5e6fdf9926f1998a2c561ac` |
| fixed, derived mass | 91 | `b419c6ec536794a5ec32951b99a55e610c7fde3d79900c4bdf929ff930c104e2` |
| public mapping | 157 | `4618dba0dc8d3e898c8930d32f28db60e1c500326802e81c1c3837d043e0ceb8` |
| exact boundaries | 43 | `350fea9db373262d29fb31dced2815c38cba4b24daf22db9bb745343ba33c09b` |

The fresh atmosphere capture produced 280 fields with fingerprint
`eab0709b975fb53c7d86bd015fc0381c2d21dc4c5ae59a11d4761acbfb8e4e19`.
Its full iteration vector is `[9, 5, 5, 5, 7, 7]`, its handoff vector is
`[8, 5, 5, 4, 6, 7]`, and its only declared deferred field is
`golden_publication`.

These digests identify the audited in-memory runs. They are not substitutes
for the five deterministic, comparison-only archives required by the exact
contract.

## Status language

- **Pass**: the causal role is necessary, current executable evidence supports
  the central claim, and a bite-sized presentation is feasible.
- **Conditional pass**: the central claim is supported, but a named diagnostic,
  test, profile, or golden must exist before the chapter may print the promised
  output.
- **Gap**: the sequence currently promises exact executable evidence that is
  not present in the audited runtime, worker, or tests.
- **Overpacked**: the material is valid but cannot all be made scientifically
  visible inside one 35-line cell without hiding an important operation.

## Cell-by-cell reconciliation

| Cell | Causal necessity and notation | Exact executable evidence now available | Verdict, remaining gap, and prose decompression |
| ---: | --- | --- | --- |
| **1. Bind chemistry to exact bytes** | Necessary. The reader must know which data define the problem before learning the encoding. \(D\), \(M\), and \(E\) are sufficient notation. | Source-data tests pin the four new asset hashes, the two catalogs' `170/23/481` and `190/23/548` extents, fixed padding, H2 `(200,)`, edge-grid members, fixture identity, and manifest ownership. The local validator enforces one-dimensional thermodynamic columns, positive finite values, increasing outer-to-inner `column_mass`, a finite nonnegative `(99,)` abundance vector, and no renormalization. | **Conditional pass.** The preflight itself can stay below 35 lines if output formatting is a helper. Add direct tests for every rejection promised in the prose: nonfinite and nonpositive temperature/pressure, a depth-dependent abundance matrix, reversed depth order, and the deliberately malformed canonical-atmosphere NPZ copy. Do not print all rejection traces in the opening; show one compact “all fail before chemistry” line after explaining the valid table. Each field still needs an explicit read/write/axis/dtype mini-contract in the generated chapter. |
| **2. Positive \(A+B\rightleftharpoons AB\)** | Necessary. It gives physical meaning to a formation constant before catalog machinery. The scalar notation is self-contained and does not compete with the exact implementation notation. | `fixed_nuclei_molecule_density`, `equal_abundance_closed_form`, and their tests establish the physical quadratic root, positivity, limiting-budget bound, exact residual closure, growth with formation constant, and growth with density scale. | **Pass.** One visible scalar function plus a four-row table is bite-sized. State explicitly that the cell uses CPU NumPy `float64`, scalar cm\(^{-3}\) densities, and a declared teaching \(K(T)\). Keep “cooler gives more molecule” conditional on the chosen constants. The missing particle row is correctly acknowledged and causally motivates Cell 5. |
| **3. Decode CO 608** | Necessary. It replaces \(A\) and \(B\) with the source-native carbon and oxygen mapping. Reusing atomic number and \(\nu_{ma}\) is correct. | The exact atmosphere catalog arrays are captured. Source-data tests compare all 170 shared records by code, component sequence, and coefficients rather than row. The teaching codec proves `608 -> (6,8)`, and the synthesis catalog's shared-record equality supports the same semantics. | **Pass with one small assertion.** Add an explicit executable CO checkpoint that reads the active component slice and asserts the two equation species are exactly 6 and 8 in each catalog. That is clearer than asking the reader to infer CO from the all-record semantic counter. Keep decimal `divmod` as explanation only; the component arrays remain authoritative. |
| **4. Three H2 policies and gates** | Necessary. Later temperature trends are uninterpretable unless the three owners and their exact gates are separated first. \(K_m(T)\) is introduced at the right point. | The atmosphere worker captures the special H2 helper, catalog-gated value, tabulated partition, 100/101/19899/19900/20000/`nextafter` probes, and the 10000/20000 state lifecycle. The synthesis worker captures the exact polynomial constants and `-700` log sentinel at 10000 K, and the public builder's provisional H2 factor/population at 9000 K. Both real boundary routes complete. | **Conditional pass.** The exact boundary claims are supported. The one-panel plotting grid is not an oracle field, but it can be evaluated locally from the staged exact functions; the printed boundary table must remain the assertion authority. Label the third curve “provisional public-builder H2 factor” so it is not mistaken for a third Newton solver. State that it uses the rounded public reference constants and is replaced by solved code 101 when molecular lines are active. Put guarded-H2 versus unguarded-malformed-polynomial overflow in a short exactness callout after the plot; it is not part of the plot's one claim. |
| **5. Smallest coupled residual** | Necessary. It is the chapter's central physical transition from conditional mass action to simultaneous conservation. The three-row notation matches the reduced exact residual. | `two_element_molecule_residual`, its analytic solution, Newton solution, conservation tests, and finite residual checks support the physical root. The production source contract confirms the signs and ownership of the corresponding rows. | **Pass.** A visible `co_residual` and three states—initial, one-ledger repair, simultaneous root—are feasible. The chapter must normalize residuals by a stated row scale rather than compare raw cm\(^{-3}\) numbers to a dimensionless tolerance. Keep the carbon-only and oxygen-only repairs as explanatory calculations, not alternative algorithms. |
| **6. Differentiate one atmosphere residual** | Necessary. It turns chemical coupling into the off-diagonal sensitivity Newton needs. The 3-by-3 derivation before a 23-row check is excellent pedagogy. | The atmosphere worker records exact returned residuals and frozen constants, but it does not serialize an exact 23-by-23 Jacobian or an independently finite-differenced production residual. The teaching tests validate only the reduced 3-by-3 Jacobian. | **Gap.** Add a narrow local exact adapter and test that return, for one declared depth and physical state, the exact `_newton_matrix_kernel` residual/Jacobian, an independently finite-differenced Jacobian, row/column labels, and absolute plus scale-relative error summaries. Do not put a 23-by-23 matrix in the notebook. The visible cell should derive and print the 3-by-3 CO block, then print one full-network error line. |
| **7. Direct solve and forced fallback** | Necessary. It makes the linear algebra boundary and singularity response explicit before positivity is discussed. No new physical notation is introduced. | The exact source contract pins `np.linalg.solve` and `np.linalg.lstsq(..., rcond=None)`, but neither atmosphere worker nor current reader test executes the forced fallback. Production captures do not prove which LAPACK branch was taken. | **Gap.** Add a controlled exact branch probe with one physical nonsingular Jacobian and one rank-deficient copy. Capture branch, rank, step norm, and \(\|J\delta-R\|\). This is a diagnostic, not a golden input. A 10–15 line visible cell is feasible once Cell 6 supplies the physical matrix. |
| **8. Atmosphere positivity branches** | Necessary. It prevents the common false statement that the atmosphere solver clips or uses logarithmic unknowns. | The atmosphere worker observes production relative updates through the exact `_newton_update_kernel`, but it does not retain per-index candidate branches, effective deltas, the mutable shared scale, or controlled witnesses for `0.69`, `abs`, `100`, and `sqrt(scale)`. | **Gap.** Add an exact controlled-vector trace and direct tests for ordinary acceptance, sign damping, absolute reflection, one-percent fallback, shared `100`, and later `sqrt(scale)=10`. The second schematic and prose should explain the order dependence before the output table. Keep this cell exclusively about the atmosphere update; synthesis's floor belongs to Cell 12. |
| **9. One-depth directions and array meanings** | Necessary. It is the first scientific closure check and the right place to distinguish raw, saved physical, transformed, and normalized arrays. The representative live-basis equation prevents a silent unit change. | The atmosphere worker provides independent 3-temperature and 3-pressure solves, iteration counts, convergence/exhaustion flags, row-scaled residual maxima, frozen and post-solve constants, raw named populations, saved physical equations, transformed equations, normalized populations, and exact 10000/20000 lifecycle probes. It also captures full shapes and the frozen `charge_square_density`. | **Conditional pass; overpacked.** Add explicit self-checks for the intended temperature/pressure directions and split particle, elemental, and charge residual summaries. The current worker stores enough data to derive those checks, but does not assert the directions. The cell's primary output should be the two control tables. Decompress the rest into three short prose steps after the table: (1) saved versus transformed units, (2) frozen constants versus post-solve population constants, and (3) code/mode dispatch. Put atom-family fallback rules in an exactness box rather than the main table. Explicitly interpret the counterintuitive captured fact that raw polynomial/H2 populations gate to zero while the later normalized-fill path can remain nonzero above the same boundary. |
| **10. Six ordered depths** | Necessary. It explains why the molecular path cannot inherit Chapter 3's depth `prange`, and it locates Numba's real benefit. The pressure-ratio notation is sufficient. | The atmosphere worker captures ordered layer indices, each actual seed, returned physical rows, iteration counts, and the one-thread environment. It therefore contains the data needed to assert pressure-scaled continuation, but its self-check does not currently make that equality explicit. It does not run the six deliberate one-depth restarts promised by the plot. | **Gap.** Add a derived exact assertion that `seed[d] == solved_physical[d-1] * P[d]/P[d-1]`, plus a separate restart trace with six iteration counts and convergence/residual fields. The plot can then consume the two iteration vectors. Static AST/source tests should pin the four `njit(cache=True,nogil=True)` kernels and the absence of `parallel=True`/`prange` in the molecular depth loop. Keep timing to one subordinate sentence; the causal claim is dependence, not speed. |
| **11. Saved seeds and molecular energy** | Necessary. It closes the atmosphere state lifecycle before changing backends. The partition-response factor and energy expression introduce only the new molecular contribution. | The atmosphere worker now captures modes 2/12, zero normalized-fill calls, raw populations, saved-row input/before/after identity, actual saved-row seed override, ordinary continuation counterfactual, unchanged normalized arrays, full energy output, direct energy reference, and convergence. The real energy route converges in one iteration at every depth from the saved physical state. | **Conditional pass; overpacked.** The central saved-versus-live-versus-restored checkpoint and one energy value are strongly supported. Add a source-level test for public-true versus lower-level-false thermal-tracking defaults and for the orchestration `finally` restoration/fallback claim before printing them. Present the visible calculation in this order: show the two array owners, run energy mode, interpret the energy. Put modes 2/12 in one small lifecycle table afterward. Put `finally`, the Chapter 3 atomic fallback, and convection-flag ownership in a short implementation note; they should not interrupt the energy derivation. |
| **12. One synthesis Newton step** | Necessary. This is the crucial backend contrast and the right point to earn \(x_e\), inverse-electron power, negative-ion corrections, `jacrev`, scaling, and the synthesis-only floor. | The synthesis route worker serializes exact final residuals, log formation constants, component multiplicities, inverse-electron powers, negative-ion flags, equation abundances, iteration counts, and nonfinite pre-replacement diagnostics for production calls. Source tests pin the direct `float32` default and the 14 shared negative-ion records. It does **not** capture a one-iteration `jacrev` result, a finite-difference Jacobian, column scaling, raw/scaled steps, damping/floor masks, a chain restart, or the order of `jacrev` and final `vmap`. The controlled nonfinite replacement test uses a constructed call rather than a full exact-source overflow route. | **Gap and the most overpacked cell.** Before drafting, add one exact single-depth step adapter with density input, residual, `jacrev` Jacobian, finite-difference Jacobian, scaling vector, fractional step, physical step, candidate, damping mask, floor mask, and residuals of the scaled and unscaled systems. Add call-order/count instrumentation proving one `jacrev` per iteration and final `vmap` only after depth loops. Add a separate exact controlled overflow witness and explicit synthesis negative-ion invariant check. Preserve one visible cell by deriving the log product and scaled system in prose first, then making the cell execute only the one-step calculation. Put chain starts, the `1e-20` network floor, direct/public dtype defaults, MPS substitution/rejection, and overflow asymmetry into compact evidence tables after the central output. Do not ask one 35-line cell to discover all of them. |
| **13. Full synthesis route** | Necessary. It establishes which of two molecular solves owns published electrons and which owns molecular arrays. The three electron labels are precise and avoid an invented “true” density. | The real full worker executes two exact calls in the required order, captures caller names, inputs, ion constants, requested device/dtype, tolerance, iteration vectors, exhaustion masks, all four outputs from each call, call-local residuals, padded arrays, and the complete public state. It asserts that call 0 supplies published \(n_e\) and call 1 supplies nuclei/molecular/equation fields. | **Conditional pass; mildly overpacked.** The CPU-float64 route claim is supported. Golden comparison is not supported until publication. Add explicit executable checks for the absent-catalog atomic-family rescaling and present-family missing-stage zero claimed in the expected output. Add a CPU-float32 full-route profile; report CUDA/MPS only when available and measured. Present the two-row call trace first, then one compact state summary. Put mixed-dtype interface behavior and accelerator profiles after the scientific ownership interpretation rather than inside the main table. |
| **14. Fixed-public-\(n_e\)** | Necessary. It corrects the common but serious inference that a fixed public field removes the internal electron equation. The input/public/internal notation is exactly what is needed. | The fixed worker performs one exact molecular solve, preserves the public input electron array bitwise, retains the internal returned electron separately, checks active/padded shapes and exhaustion, and supports both derived and supplied `mass_density` arguments. The real default fixed route completes. | **Conditional pass.** Run and retain a real supplied-mass capture as well as the derived-mass capture in the publication workflow, and declare where the supplied positive mass vector comes from. At present the worker supports the branch, but the standard CLI and focused tests exercise the real derived branch only. Golden comparison is also pending. The visible code can remain small: two exact calls, one branch table, one electron-identity assertion. |
| **15. Route ownership matrix** | Necessary. After the full and fixed routes have been earned, a matrix is the clearest way to prevent incompatible claims from being blended. No new algebra is appropriate. | The atmosphere worker now executes and asserts the full, structured-handoff, pressure-disabled, bridge-H2 mixed/all-zero, live shape-error, modes, energy, and structural-Doppler routes. The synthesis public worker asserts one fixed EOS solve, returned-array reuse, no fallback re-solve, no-ground partitions, one edge-grid load, and exact builder ownership. Full/fixed synthesis workers provide the other matrix rows. | **Conditional pass; heavily overpacked.** The six core route rows are supported. Missing evidence remains for the exact zero-code priming/cached 230-job schedule: the atmosphere worker proves one route-level solve and one fill, but does not trace the priming call followed by every cached population job. Add that call trace. The cell should print the ownership matrix as its only main output. Place H2 global selection, the live `(200,)`/`(D,170)` failure, and slots 919/927 in three short post-matrix exactness notes. Cell 15 should own “one solve then cached reuse” but not repeat the 230-job composition and slot inventory that Cell 17 owns. |
| **16. Align catalogs by semantics** | Necessary. It separates chemistry-record identity from row position before public mapping. No new physical notation is introduced. | Source-data tests key every active record by rounded code, exact component sequence, and exact coefficient tuple. They assert 170 shared, zero atmosphere-only, the exact 20 synthesis-only codes, different row order, active bounds, true zero padding, atmosphere coefficient row 6, and reverse-lookup `-1`/electron/one-past-active semantics. | **Pass.** This is executable and bite-sized if comparison logic lives in a transparent helper and the visible cell prints counts, the 20 codes, and mismatch count. Do not display all 170 records. Preserve the qualification that this pinned pair is a strict code-semantic extension, not a general future file-format promise. |
| **17. Public molecular lane** | Necessary. It completes the chapter by identifying the exact normalized quantity and public axis needed by Chapter 5. Calling stage index 5 a synthetic lane rather than a sixth ion stage is essential notation. | The real public worker proves all 54 species mappings, `276 -> 608 -> column 45`, one fixed molecular solve, reuse of its equation and molecular arrays, no-ground partitions, stage-5 writes, unchanged unowned partition cells, unchanged ion cube, solved code-101 H2, exact edge arrays, and schema-v4 fields. Source tests pin the hard-coded `(99,)` molecular-mass hash. | **Conditional pass; overpacked.** Add an algebraically independent CO \(N_{\rm CO}/U_{\rm CO}\) checkpoint; the current worker compares production mapping to another call of the same exact line-population helper, so it does not yet supply the independently calculated normalization value promised by the sequence. Add the atmosphere 230-job composition and pinned-slot assertion here, but do not repeat its solve/cache count from Cell 15. Print the actual 54 destination columns or their compact ranges so readers do not infer that molecules occupy only columns 99–138; all mappings use stage 5, but their columns include atomic-number-like positions such as CO's 45. Keep the visible cell to one CO trace plus a vectorized all-54 assertion. Put the 16 legacy atmosphere mappings and the Chapter 5 field inventory into separate post-result tables. Golden comparison remains pending. |

## Missing executable fields and assertions

The following list distinguishes data that are genuinely absent from data that
already exist but are not yet asserted.

### P0: add before Chapter 4 prose claims exact executable parity

1. **Atmosphere production Jacobian checkpoint for Cell 6**

   Required result fields:

   - declared depth, physical density input, and frozen formation constants;
   - exact 23-row residual and 23-by-23 Jacobian;
   - independent finite-difference Jacobian;
   - row/column species codes;
   - maximum absolute and scale-relative differences;
   - extracted C/O/particle 3-by-3 block.

2. **Atmosphere linear-branch checkpoint for Cell 7**

   Required fields:

   - direct-solve and forced-fallback branch labels;
   - matrix ranks;
   - exact steps;
   - step norms and linear residual norms.

3. **Atmosphere controlled positivity trace for Cell 8**

   Required fields:

   - old density, raw delta, previous delta, effective delta;
   - candidate, returned density, branch mask, and shared scale by index;
   - explicit witnesses for `0.69`, reflection, one-percent fallback,
     `100`, and later `10`.

4. **Continuation-versus-restart evidence for Cell 10**

   The current atmosphere worker already stores `full_layer_seed_equation_densities`
   and saved physical solution rows. Add a self-check of their exact
   pressure-scaled equality rather than duplicate them in a new archive
   field. Add the genuinely absent restart iteration, convergence, exhaustion,
   and residual vectors.

5. **Synthesis exact one-step checkpoint for Cell 12**

   Required fields:

   - one physical density vector and its units/dtype/device;
   - exact residual and `jacrev` Jacobian;
   - independent finite-difference Jacobian;
   - density-column scaling vector;
   - fractional and physical steps;
   - unscaled and scaled linear residuals;
   - sign-flip mask, `0.69`-damped step, candidate, and floor mask;
   - one explicit `chain_length` restart seed;
   - call-order/count evidence for per-iteration `jacrev` and post-loop `vmap`;
   - an exact-source nonfinite-product replacement witness distinct from the
     healthy production result.

6. **Route-level fixed supplied-mass result for Cell 14**

   Retain both real fixed-route branches in the deterministic publication
   input. The supplied vector must have a declared input owner and exact
   identity.

7. **Atmosphere zero-code priming and 230-job cache trace**

   Capture:

   - one zero-code priming call;
   - one molecular solve for the temperature index;
   - all 230 subsequent jobs;
   - zero additional molecular solves;
   - 198 atomic jobs and 32 molecular jobs;
   - the 16 codes, modes 1/11, and pinned packed destinations.

8. **Independent public CO normalization**

   Reconstruct one CO \(N/U\) value from the captured equation densities,
   no-ground neutral partitions, exact component mapping, and the hard-coded
   molecular masses without calling
   `molecular_line_populations_by_species_code` a second time. Compare it to
   `[depth,5,45]`.

9. **Comparison-only golden publication**

   The atmosphere worker explicitly leaves only `golden_publication`
   deferred, and all synthesis workers currently build in memory only. Cells
   13, 14, and 17 must not say “matches the golden” until the five archives
   have been built twice in fresh processes, compared byte-for-byte, and
   published through the deterministic writer.

### P1: add before final chapter acceptance

1. Assert the physical temperature and pressure directions from the stored
   independent atmosphere controls. Report the particle, elemental, and
   charge rows separately rather than only a maximum residual.
2. Add direct wrapper tests for every rejection promised by Cell 1, including
   malformed canonical input structure and depth-dependent abundances.
3. Add source/runtime tests for:

   - all synthesis negative-ion ordering and sign corrections;
   - the four exact atmosphere molecular `njit` kernels and lack of molecular
     `prange`;
   - public-true versus lower-level-false thermal-tracking defaults;
   - orchestration restoration and atomic-energy fallback;
   - absent-family atomic rescaling and present-family missing-stage zero;
   - direct/public dtype differences, CPU float32, and MPS
     substitution/rejection. CUDA/MPS numerical claims remain conditional on
     availability.

4. Serialize or generate an explicit field contract for every visible cell:
   reads, writes, units, axes, dtype, device, and whether the output is a
   candidate calculation or a comparison-only oracle. The sequence's global
   checklist requires this, but most individual cell descriptions do not yet
   state all seven items.
5. Attach schema axes and units to the Chapter 5 handoff table. A correct key
   set and shape are not, by themselves, an explanation of an array.

## Redundancy reconciliation

Only three repetitions need active control.

### Formation policies

Cell 4 owns every formula and exact temperature gate. Cell 9 may show the
resulting raw/normalized boundary behavior, and Cell 17 may name the
normalization policy, but neither should repeat the polynomial.

### Route ownership

Cells 13 and 14 own the calculations for the full and fixed synthesis routes.
Cell 15 owns their comparison and adds only the routes not yet executed. It
must not reprint full state arrays.

### Atmosphere molecular schedule

The current sequence mentions the 230-job schedule in both Cells 15 and 17.
Assign ownership as follows:

- Cell 15: one priming solve and subsequent cache reuse;
- Cell 17: the exact `198 + 2 x 16 = 230` composition, codes, modes, and packed
  destinations.

This preserves all material while removing duplicate explanation.

## Prose decompression plan

The following pattern keeps the 17-cell count and the 35-line limit without
hiding scientific work:

1. **Before the cell:** explain the physical question, derive the new equation,
   and name the array units.
2. **In the cell:** perform exactly one scientific action. Formatting,
   immutable loading, and call tracing may be helpers; residual construction,
   a Newton update, or a normalization identity may not be hidden.
3. **Immediately after:** interpret one table or one-panel plot.
4. **Then:** place secondary source seams in a short “exact implementation
   note,” tied to a passing assertion.
5. **Finally:** state the one earned contract and causal next link.

Apply this most strongly to:

- **Cell 9:** control tables first; lifecycle and dispatch afterward;
- **Cell 11:** saved-state checkpoint first; modes and orchestration afterward;
- **Cell 12:** one Newton step first; dtype, sentinel, restart, and overflow
  tables afterward;
- **Cell 13:** two-call ownership first; profiles and fallback families
  afterward;
- **Cell 15:** route matrix first; bridge, disabled-state, and Doppler notes
  afterward;
- **Cell 17:** CO and all-54 lane check first; legacy schedule and handoff
  inventory afterward.

No detached “try it yourself,” exercise, homework, or problem section should
be added. Any useful diagnostic remains in the main causal text.

## Notation and interpretation corrections

1. Keep every Newton unknown in cm\(^{-3}\) density space. Use “log-product
   evaluation” only for synthesis molecular products.
2. In Cell 4, distinguish the public builder's provisional H2 formation factor
   from the solved code-101 population and from the atmosphere table policy.
3. In Cell 9, explicitly say that transformed atmosphere equation columns are
   no longer physical cm\(^{-3}\) densities. The saved rows remain the warm
   start authority.
4. In Cells 13-15, never use an unqualified “electron density.” Always name
   input, internal, or public ownership.
5. In Cell 17, print `stage=5` and `column=species_code//6-1` as separate
   identifiers. The synthetic stage lane is not a sixth ion stage, and its 54
   destination columns are not one contiguous molecule-only block.
6. Interpret structural `+inf` Doppler widths as an exact support-array
   exception at zero-population, zero-mass slots 919 and 927. Do not present
   them as a physical infinite broadening and do not sanitize them.

## Prioritized recommendations

### P0 — before drafting the 17 visible cells

1. Implement and test the missing exact kernel checkpoints for Cells 6-8, 10,
   and 12.
2. Add the independent CO normalization checkpoint and the atmosphere
   priming/cache schedule trace.
3. Capture the real supplied-mass fixed route.
4. Build and double-capture the five comparison-only goldens; until then,
   remove all golden-match language from generated prose.
5. Freeze one explicit read/write/units/axes/dtype/device contract for each
   cell.

### P1 — during chapter generation

1. Use the prose decompression plan without increasing the chapter or visible
   cell count.
2. Make the two quantitative plots only after their deterministic tables pass.
3. Keep Cell 15's matrix and Cell 17's mapping table compact; do not print
   whole source arrays or source blocks.
4. Add the CPU-float32 profile and conditional accelerator policy checks.
5. Audit the Chapter 3 neighbor wording so it does not imply that all
   molecular values live only in public columns 99–138.

### P2 — final zoom-out audit

1. Trace every cell to exactly one arrow in the global causal spine.
2. Remove repeated formulas, especially H2 polynomials and route definitions.
3. Check every expected output is interpreted before the next abstraction.
4. Confirm the closing summary adds no new number, function, exception, or
   limitation.
5. Retain the causal Chapter 5 link and do not reopen molecular equilibrium in
   Chapter 5.

## Acceptance decision

The 15-chapter architecture and Chapter 4's three-movement, 17-cell structure
should be retained. Cells 2, 3, 5, and 16 are ready to draft from current
evidence. Cells 1, 4, 9, 11, 13, 14, 15, and 17 are ready only after their
named conditional evidence or prose decompression is supplied. Cells 6, 7, 8,
10, and 12 remain blocked on exact executable diagnostics.

The chapter should not be accepted, and its parity language should not be
published, until the P0 gaps and five-golden publication gate are closed.
