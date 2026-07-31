# Chapter 3 Central Integration Plan

This file is the root-agent reconciliation of:

- `design/chapter03_exact_source_contract.md`;
- `design/chapter01_02_03_continuity_audit.md`;
- `design/chapter03_causal_outline.md`;
- the staged progressive source and role-split data layer;
- the focused audit of `book/chapter03_support.py`.

It freezes the implementation order for canonical Chapter 3. If a draft
conflicts with this file, the Bible and exact-source contract still have higher
authority.

## Chapter identity

**Title:** Atoms, Ions, and Electrons

**Central question:** Given temperature, material gas pressure, and a linear
elemental mixture, which atomic levels and ion stages are occupied, and which
electron density makes those populations charge-consistent?

**Claim to earn:** In an atom-only LTE gas, level weights and Boltzmann factors
build each partition function, adjacent-stage Saha ratios connect ion stages,
and a damped charge-conservation fixed point closes the electron density.

**Chapter 4 dependency:** Molecules spend several elemental budgets at once,
so the element-by-element atomic closure is no longer separable.

There are no detached exercises. A useful prediction, limiting case, invalid
input, or parameter variation is resolved where it becomes causally useful and
is interpreted immediately.

## Frozen opening order

The opening must not contain an unexplained ion-fraction preview.

1. One sentence inherits Chapter 1's `temperature` and `gas_pressure` and
   Chapter 2's declared linear mixture and empty population fields.
2. Compute only the ideal-gas particle budget
   \(P_{\rm gas}/(k_{\rm B}T)\).
3. Show, conceptually rather than as equilibrium output, that several
   neutral/ion/electron allocations can share that budget.
4. State the central question and claim.
5. State atom-only LTE, ideal-gas, local-depth, CGS, plane-parallel/static
   assumptions; molecules are disabled because they are the next dependency,
   not because every target atmosphere is warm.
6. Show two separate exact engine contracts:

| route | reads | writes | execution |
| --- | --- | --- | --- |
| atmosphere | `temperature (D,)`, `gas_pressure (D,)`, `elemental_abundances_by_layer (D,99)`, positive `electron_density (D,)` seed | density fields and two sparse `(D,1006)` population arrays | NumPy `float64`, CPU/Numba |
| synthesis | path-accepted `temperature`, `gas_pressure`, linear `elemental_abundances`, and manifest-bound `EOSTables` | internal `(D,99,6)` and public `(D,6,139)` atomic states | selected Torch dtype/device plus declared host-`float64` decision islands |

`gas_pressure`, not Chapter 1's `total_pressure`, enters the EOS. A
one-dimensional abundance vector is broadcast only by the exact boundary that
accepts it; the text must not invent one universal abundance input shape.

7. Place the conceptual levels → partition → ion ladder ↔ charge schematic.
8. Define a level, excitation energy, and statistical weight; derive and
   execute the two-level Boltzmann ratio.
9. At that first numerical cell, state explicitly that Chapter 1 used exact
   constants for Planck radiation while this EOS calculation keeps the rounded
   reference literals required for numerical identity.
10. Build \(U\), derive \(n/U\), and verify
    \(n=U(n/U)\). Chapter 2 owned the representation contract; Chapter 3 owns
    its physical derivation.

## Frozen notation fence

The text must distinguish physical charge, spectroscopic label, and zero-based
storage:

| physical species | charge \(q\) | spectroscopic label | public axis index |
| --- | ---: | --- | ---: |
| neutral hydrogen | 0 | H I | 0 |
| singly ionized hydrogen | 1 | H II | 1 |

Use \(r=q=0,1,\ldots\) for the mathematical ladder. If an exact helper takes a
one-based `ion_stage`, identify that convention beside the call. Six stages is
storage capacity, not a universal physical stage count.

The exact representation meanings are:

```text
n_{s,r}             → ion_stage_populations
n_{s,r} / U_{s,r}   → partition_normalized_populations
Δv_D / c             → fractional_doppler_widths
```

`partition_normalized_populations` is never called a fraction, an actual stage
population, or a bound-level population.

## Three-movement narrative

### Movement I — Count states at one depth

The remembered question is: what do temperature and electron crowding do to
one atom?

1. two-level Boltzmann ratio;
2. partition normalization and the physical meaning of \(n/U\);
3. three production partition recipes and their separate stack policies;
4. Saha's adjacent-stage ratio and its limiting behavior;
5. pressure lowering and occupation gates.

Partition recipes are taught once. The density-correction section may point
back to them but may not repeat their taxonomy. Ordered production partition
sums must not be called log-stable; log-space stability belongs to the
synthesis Saha ladder and the labeled teaching helper.

Natural pause: a supplied \(n_e\) now produces density-corrected atomic stage
populations, but the supplied density has not been charge-validated.

### Movement II — Close the atomic gas

The remembered question is: which electron density is unchanged by the
population-and-charge update?

1. hydrogen-only damped fixed point with its 75-percent one-step floor;
2. positive-state precondition and impossible-seed failure;
3. ordered 99-element retained-stage charge sum;
4. full one-depth atmosphere closure and refreshed populations;
5. independent atom-only depths through the real `prange` boundary;
6. density, atom-only collision proxy, path-specific Doppler support;
7. atomic specific internal energy.

The solver is a damped fixed point, never Newton. The atmosphere refresh after
its final update and the synthesis full solver's one-evaluation-behind result
must both be stated. Generic Numba syntax, cache behavior, and device selection
belong to Chapter 2 and are not retaught.

Natural pause: the chapter has a charge-closed, multi-depth atom-only state
with the support quantities downstream physics will consume.

### Movement III — Cross the exact engine boundary

The remembered question is: how can the same physical values move between
different layouts without changing their meaning or claim?

1. packed atmosphere schedule and sparse `(D,1006)` layout;
2. internal synthesis `(D,99,6)` axes;
3. public synthesis `(D,6,139)` axes;
4. sentinel routing before physical routing;
5. full closure versus fixed-\(n_e\) comparison;
6. backend-specific parity status.

The main fixed-\(n_e\) comparison must receive the electron density produced by
the full atom-only closure unchanged. A perturbed density may appear only as an
ownership/failure demonstration. The fixed route preserves its input and makes
no charge-closure claim.

Natural close: atomic budgets are trustworthy and correctly represented, but
molecules couple the budgets of several elements.

## Visible evidence budget

The post-audit target is 21 substantial executable cells, each at most 35
lines, after splitting three overloaded operations and making the internal
energy derivation executable:

| cells | purpose |
| --- | --- |
| C1–C2 | Boltzmann ratio; partition normalization and \(n/U\) reconstruction |
| C3–C6 | transparent fixed-\(n_e\) Saha; production partition preflight; exact modes; pressure lowering |
| C7–C10 | scalar fixed point; 99-element closure; timing; separate all-field parity |
| C11–C15 | mass/collision support; Doppler limits; packed support; excitation-energy identity; full energy ledger |
| C16–C18 | packed schedule; sentinel map; physical fixed-density map |
| C19–C21 | full/fixed claims; separate abundance-once probe; backend tolerance profile |

At most three short exact-source cards are allowed:

- one ordered ground-partition definition if output alone is insufficient;
- the real depth-`prange` ownership boundary;
- the packed decoder.

If all three are not necessary, omit them. Production-sized code never appears
inside Markdown.

Every visible cell has a prediction, an explicit reads/writes/unit/shape/
dtype/device contract, visible output, immediate interpretation using actual
values, and a prose bridge to the next dependency.

## Visual contract

Exactly three required one-panel quantitative figures:

1. excited/ground population ratio versus temperature;
2. H I/H II fractions versus temperature at declared fixed \(n_e\);
3. fixed-point residual versus iteration.

Exactly two original textbook schematics:

1. levels and weights → \(U\) → ion ladder ↔ charge closure;
2. sparse `(D,1006)` state → explicit map → `(D,6,139)` cube.

Both schematics use newly owned prompts and website-inspired visual language,
not official compositions or assets. Plots use the common book style, one
panel and one claim, direct labels where clearer, and actual-output
interpretation in the following paragraph.

## Progressive source and data boundary

The staged exact layer includes:

- atmosphere constants, EOS, population layout, runtime state, Doppler,
  atomic specific energy, and structured synthesis bridge;
- synthesis EOS, ground partition table, paths, and the exact progressive
  `load_atomic_masses` and `compute_doppler_per_ion` definitions;
- five atmosphere table bundles and the synthesis atomic-mass table;
- a deterministic split of the formerly mixed synthesis EOS bundle into a
  39-array static table archive and an 11-array depth fixture.

The exact files and displayed definitions must continue to pass
`scripts/verify_pinned_source_fragments.py`. Notebook setup must establish
`PAYNE_ZERO_DATA_ROOT` before importing exact atmosphere runtime modules.
Fixed-\(n_e\) synthesis calls must provide the manifest-bound atomic-mass table
or an explicit mean/mass input.

The reader runtime imports only this repository. The pinned Payne Zero checkout
is a development oracle and golden-output producer, never a notebook
dependency.

## Golden-output build order

Goldens may be produced only after the local fixture and exact call boundary are
frozen. Builders must use the pinned checkout read-only, write only into this
repository, and record source commit, input hashes, array hashes, shapes,
dtypes, units, and builder command.

Build and consume in this order:

1. `chapter03_atmosphere_saha_outputs.npz` after local scalar/batch mode
   evaluation;
2. `chapter03_atmosphere_atomic_state.npz` after local one-depth and depth
   closure;
3. energy fields inside the atmosphere-state golden after local energy
   evaluation;
4. `chapter03_packed_bridge_outputs.npz` after local sentinel and physical
   mapping;
5. `chapter03_synthesis_atomic_state_cpu_float64.npz` after local full/fixed
   evaluation;
6. optional backend profiles only on actually available hardware.

Goldens are loaded after computation, never used to generate the displayed
answer.

## Implementation slices

1. Close focused teaching-helper audit gaps and keep helpers private to the
   book.
2. Finish source/data staging verification and record exact import
   preconditions.
3. Build the minimal Chapter 3 integration fixture and oracle builders.
4. Add deterministic goldens and focused parity tests.
5. Add both schematic specifications, generate, inspect, and hash the assets.
6. Write canonical `book/chapters/chapter_03.py` in the three movements.
7. Execute and render the notebook; inspect every output, plot, schematic,
   movement boundary, summary, and next link in the local reader.
8. Run chapter-level exactness, pedagogy, visible-cell, source-dump, and
   no-exercise tests.
9. Reread the end of Chapter 2, all of Chapter 3, and the planned opening of
   Chapter 4 as one narrative.
10. Accept Chapter 3 only after all P0/P1 findings are closed.

## Acceptance failures

The chapter fails if it contains any of the following:

- an ion-fraction preview before Saha is derived;
- a repeat of Chapter 2's schema inventory, dex lesson, checksum tamper
  demonstration, Numba syntax lesson, device discovery, or cache timing;
- `total_pressure` at an EOS boundary;
- logarithmic abundance supplied to a linear-abundance boundary;
- an ambiguous “stage 1”;
- a silent change from exact radiation constants to rounded EOS constants;
- a fixed-\(n_e\) state credited with charge closure;
- a raw reshape between `(D,1006)` and `(D,6,139)`;
- equality claims between the atmosphere and synthesis EOS stacks;
- molecule populations, molecular mass action, or coupled continuation;
- a detached exercise section;
- a summary that introduces a new field, calculation, species, or result.

The final summary must answer the opening question, enumerate only demonstrated
outputs, state the full/fixed claim distinction, name what remains
unavailable, and link causally to Chapter 4.
