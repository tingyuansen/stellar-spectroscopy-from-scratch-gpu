# Chapter 1→2→3 Continuity, Notation, Redundancy, and Handoff Audit

Audit date: 2026-07-30

Scope: independent neighbor audit of `BIBLE.md`,
`design/pedagogical_flow_rubric.md`,
`design/global_chapter_contracts.md`, the canonical Chapter 1 and Chapter 2
sources, `design/chapter02_acceptance.md`, and
`design/chapter03_exact_source_contract.md`.

This audit does not authorize changes to canonical chapter, source, or data
files. Its purpose is to constrain the Chapter 3 draft before implementation.

## Outcome

There is no P0 continuity blocker. Chapters 1 and 2 form a coherent earned
sequence:

1. Chapter 1 constructs an explicitly incomplete depth, temperature, column
   mass, and material-pressure scaffold.
2. Chapter 2 makes numerical meaning, acceleration, data identity, abundance
   representation, and the structured boundary trustworthy.
3. Chapter 3 can therefore begin directly with the missing physical question:
   how the declared mixture is divided among atomic levels and ion stages, and
   which electron density closes charge.

The Chapter 2 two-movement decision helps this handoff rather than harming it.
Movement I earns the computational discipline that Chapter 3 may now apply;
Movement II moves from generic numerical meaning to the mixture and population
fields that Chapter 3 must fill. The Chapter 2 close names exactly three
unearned values—`electron_density`, `ion_stage_populations`, and
`partition_normalized_populations`—and Chapter 3's physical scope answers that
promise directly
(`book/chapters/chapter_02.py:1653-1665`).

The main risk is not a missing topic. It is repetition. The Chapter 3 source
contract is scientifically rich enough to become another very dense chapter,
and several proposed moves would re-stage demonstrations that Chapter 2 has
already completed. The precise opening and handoff contracts below prevent
that drift.

## What the reader has already earned

| Earned object or idea | Evidence | Chapter 3 consequence |
| --- | --- | --- |
| A one-dimensional, plane-parallel, static atmosphere ordered from the outermost layer inward | `book/chapters/chapter_01.py:134-166`; `BIBLE.md:222-225` | State the inherited assumptions once; do not re-teach the geometry. |
| CPU NumPy `float64` arrays `temperature`, `column_mass`, `gas_pressure`, and `standard_rosseland_optical_depth` on the standard 80-layer grid | `book/chapters/chapter_01.py:36-51`, `1193-1224`; `design/global_chapter_contracts.md:153-180` | The names, units, depth direction, and controlled-scaffold status may be assumed. The arrays must not be called a converged atmosphere. |
| `gas_pressure` is the material pressure after the radiation-pressure contribution has been separated from `P_total` | `book/chapters/chapter_01.py:995-1065` | The ideal-gas EOS must use `gas_pressure`, never the Chapter 1 `total_pressure` intermediate. One brief reminder is warranted because both quantities appeared in Chapter 1. |
| LTE as a local thermodynamic assumption, not a claim that the whole emergent spectrum is one blackbody | `book/chapters/chapter_01.py:566-646` | Chapter 3 may invoke LTE but must newly derive Boltzmann and Saha statistics. It should not re-derive Kirchhoff's law or the source function. |
| Every numerical boundary requires meaning, units, axes, dtype/device, data identity, and a validation claim | `book/chapters/chapter_02.py:245-250`; `design/chapter02_acceptance.md:9-20` | Begin with one compact EOS reads/writes contract; do not teach “what is a tensor/dtype/device” again. |
| Numba, `njit`, `parallel=True`, `prange`, private ownership, and reduction order | `book/chapters/chapter_02.py:506-620` | Show only the new physical consequence: one depth owns one whole fixed-point orbit and requires no cross-depth reduction. |
| Torch device policy and CPU/CUDA/MPS dtype policy | `book/chapters/chapter_02.py:675-757` | State the selected backend beside synthesis arrays; do not repeat device discovery or a generic backend tutorial. |
| Element/isotope/ion/molecule distinctions and standard, direct, and deck abundance representations | `book/chapters/chapter_02.py:1022-1246`; `design/global_chapter_contracts.md:188-194` | Consume a declared **linear** 99-element number-abundance vector/matrix. Do not repeat dex arithmetic, alpha-element lists, quantization, or deck decoding. |
| Four data roles, manifests, SHA-256 identity, and golden-after-computation policy | `book/chapters/chapter_02.py:1248-1410` | Chapter 3 may run one concise table preflight and later load goldens. It must not repeat the checksum lesson or byte-tamper demonstration. |
| A schema is structural rather than physical validation | `book/chapters/chapter_02.py:1467-1613` | Do not repeat the 25-field inventory or schema failure injection. The Chapter 2 schema fixture is interface-only and must not become Chapter 3's physical EOS input. |
| Operational distinction between actual stage density \(n\) and stored \(n/U\), despite identical public shape | `book/chapters/chapter_02.py:1412-1464` | Chapter 3 owns the physical derivation of \(U\), not another same-shape or \(10^{12}/4\) demonstration. |
| Conceptual distinction between full charge closure and fixed-\(n_e\) population filling | `book/chapters/chapter_02.py:1615-1622` | Chapter 3 must execute and measure both routes. It should introduce neither name as if unseen. |

## What Chapter 3 must still teach

The following objects remain wholly unearned and therefore need the full
question → intuition → mathematics → bite-size code → interpreted result →
production-boundary treatment:

- atomic level, excitation energy, statistical weight, and level population;
- the Boltzmann ratio and the partition sum \(U\);
- spectroscopic ion stage versus electric charge versus array-axis index;
- adjacent-stage Saha equilibrium and its limiting behavior;
- production partition branches, their exact/reference constants, and the
  stack-specific ground/occupation policies;
- self-consistent charge closure as a damped fixed point;
- total nuclei number density, `mass_density`, the neutral-collision proxy,
  fractional Doppler support, and atomic specific internal energy;
- the sparse atmosphere `(D,1006)` layout, synthesis-internal `(D,99,6)`
  layout, and public `(D,6,139)` layout;
- measured atmosphere scalar/depth-parallel behavior and measured synthesis
  full-closure/fixed-\(n_e\) behavior.

Molecular equilibrium, molecular internal energy, continuum opacity, bound-line
opacity, damping, transfer, and atmosphere convergence remain later
dependencies. A support quantity may be produced here, but its later physical
use should receive only one causal sentence.

## Priority findings

### P0

None.

### P1. The proposed opening numerical preview spends an answer before it is earned

The proposed Section 3.1 asks for “one single-panel ion-fraction preview
without yet claiming how it was computed”
(`design/chapter03_exact_source_contract.md:1646-1651`). That conflicts with
the Chapter 2 close, which explicitly says those values have not been earned
(`book/chapters/chapter_02.py:1660-1665`), and with the book's code-as-evidence
rule. A numerical curve with an undisclosed computation is neither a
conceptual schematic nor an interpretable calculation.

**Required resolution.** Open with a non-uniqueness argument that needs no
hidden EOS result: \(P_{\rm gas}/(kT)\) supplies a total particle budget, but
several allocations among neutral atoms, ions, and electrons can share that
budget. Use the first numerical ion-fraction plot only after the Saha ratio has
been derived and executed. The first schematic may preview the dependency
loop, but it must be explicitly conceptual.

### P1. The Chapter 2/3 ownership seam for \(n\) versus \(n/U\) needs one explicit rule

Chapter 2 already gives the equal-shape, equal-unit, different-meaning example
for `ion_stage_populations` and `partition_normalized_populations`
(`book/chapters/chapter_02.py:1412-1464`). The Chapter 3 plan proposes again to
“show why the reusable quantity is \(n/U\)” and to delay the exact names until
the distinction is constructed
(`design/chapter03_exact_source_contract.md:1671-1678`). As a chapter-local
outline that would read as a repeated first introduction.

**Required resolution.** Treat Chapter 2 as having earned the *representation
contract* and Chapter 3 as owning the *physical derivation*. Recap in one
sentence: “Chapter 2 named the two arrays but supplied \(U=4\); we can now
compute \(U\) from actual levels.” Then derive \(U\), calculate \(n/U\), and
verify \(n=U(n/U)\). Do not repeat the \(10^{12}/4\) arithmetic, equal-shape
fixture inspection, or same-shape schematic. The Chapter 3 layout schematic
must teach sparse-slot mapping, not re-teach representation ambiguity.

### P1. The opening state contract must separate the two exact engine states

The proposed Section 3.2 says only to state “CPU atmosphere state, Torch
synthesis state, and exact input/output shapes”
(`design/chapter03_exact_source_contract.md:1655-1661`). That is too compact
for the first chapter in which “composition” becomes an executable EOS input.
Chapter 2 has already shown that three abundance arrays can share shapes while
holding different quantities.

**Required resolution.** Before the first production call, show two rows:

| path | reads | writes | dtype/device |
| --- | --- | --- | --- |
| atmosphere | `temperature (D,)` K, `gas_pressure (D,)` dyn cm\(^{-2}\), `elemental_abundances_by_layer (D,99)` linear relative number abundance, positive `electron_density (D,)` seed in cm\(^{-3}\) | density fields and two sparse `(D,1006)` population arrays | NumPy `float64`, CPU |
| synthesis | the same physical columns with exact `elemental_abundances` shape accepted by the called boundary, plus manifest-bound `EOSTables` | internal `(D,99,6)` tensors and public `(D,6,139)` arrays | selected Torch dtype/device, with declared host `float64` decision islands |

Also say explicitly that the ideal-gas particle budget uses the Chapter 1
`gas_pressure`, not `total_pressure`, and that any one-dimensional abundance
vector is broadcast into a depth-dependent mixture only by the exact called
boundary. Do not invent a universal input shape.

### P1. Ion-stage notation is not yet safe enough for a first-time reader

The source contract uses \(r\) in \(n_{s,r}\), \(q_r\) in charge closure, a
public six-slot stage axis, spectroscopic labels H I/H II, and source APIs whose
`ion_stage` arguments can be one-based
(`design/chapter03_exact_source_contract.md:199-256`,
`1228-1260`). Without a notation fence, “stage 1” can mean neutral
spectroscopic stage I, array index 1, or charge \(+1\).

**Required resolution.** Before Saha, include a compact exact map:

| physical species | charge \(q\) | spectroscopic label | public axis index |
| --- | ---: | --- | ---: |
| neutral H | 0 | H I | 0 |
| singly ionized H | 1 | H II | 1 |

Use \(r=q=0,1,\ldots\) in the mathematical ladder unless reproducing an exact
source signature. Whenever an exact helper uses a one-based `ion_stage`, say so
beside that call. “Six” must be called storage capacity, not a universal
number of physical stages.

### P1. Chapter 1's exact constants and Chapter 3's rounded EOS constants need an explicit seam

Chapter 1 introduces exact \(h\), \(c\), and \(k_{\rm B}\) values in the Planck
calculation. Chapter 3's production EOS instead uses parity-pinned rounded
reference values such as `1.38054e-16` and `8.6171e-5`
(`design/chapter03_exact_source_contract.md:73-88`). The proposed flow waits
until Section 3.5 to explain exact/reference constants
(`design/chapter03_exact_source_contract.md:1680-1695`), after the two-level
Boltzmann calculation has already used \(k_{\rm B}\).

**Required resolution.** At the first numerical Boltzmann cell, state which
constant tier is being used. Prefer the EOS reference constant if the values
will later be compared with production output. Explain once that Chapter 1
used the exact physical tier for Planck radiation, while the EOS keeps rounded
reference literals for numerical identity. Do not create a shared “cleaned”
constant namespace.

### P1. The fixed-\(n_e\) comparison must preserve Chapter 2's upstream-ownership claim

Chapter 2 defines fixed \(n_e\) as accepted from the upstream physical
atmosphere, not arbitrarily chosen
(`book/chapters/chapter_02.py:1615-1622`). The Chapter 3 plan says to run full
and fixed paths from “the same controlled columns” and “the same supplied
density” without specifying how that density was earned
(`design/chapter03_exact_source_contract.md:1773-1782`).

**Required resolution.** In the main comparison, run full atomic charge closure
first and pass its resulting electron-density column unchanged to the
fixed-\(n_e\) route. The fixed route must reproduce that input exactly and make
no closure claim. A deliberately perturbed supplied density may then show that
the bridge preserves even an inconsistent input, but it must be labeled a
failure/ownership demonstration rather than an accepted atmosphere. The
Chapter 3 controlled fixture itself remains a chapter calculation, not a
physically converged atmosphere product.

### P1. The planned chapter needs internal movements before prose and code are drafted

The contract currently describes fifteen numbered movements spanning
partition taxonomy, pressure ionization, two closure stacks, support
quantities, three layouts, and two bridge claims
(`design/chapter03_exact_source_contract.md:1641-1801`). Chapter 2 already
required 31 visible code cells and an explicit mid-chapter pause
(`design/chapter02_acceptance.md:70-84`). Chapter 3 can easily exceed the same
retention limit even without redundant tutorials.

**Required resolution.** Keep one navigation chapter for now, but draft it
with three visible movements:

1. **Movement I — Count states at one depth:** level → Boltzmann → \(U\) →
   production partition branches → Saha and pressure lowering.
2. **Movement II — Close the atomic gas:** one-element feedback → 99-element
   charge closure → independent depths → density, Doppler, and energy support.
3. **Movement III — Cross the exact engine boundary:** packed/public layouts
   → full-closure versus fixed-\(n_e\) claims → summary and molecular handoff.

Each movement needs one remembered question and a natural pause. Do not split
to a sixteenth chapter before measuring substantial visible cells and rendered
pacing, but do not wait until after a 30-cell draft to create the pauses.

### P1. “Stable level sums” must not be presented as the production partition algorithm

The global Chapter 3 ownership phrase includes “stable level sums”
(`design/global_chapter_contracts.md:235-240`), but the exact source contract
states that special partition functions are direct ordered additions and that
only the synthesis Saha ladder is log-stable
(`design/chapter03_exact_source_contract.md:258-279`,
`1567-1569`). This is a parity-sensitive distinction.

**Required resolution.** A stable classroom helper may be shown as a numerical
idea, but it must not be called the exact partition implementation. Production
source cards and parity checks must retain direct ordered summation. Reserve
the log-space description for the synthesis Saha ladder.

### P2. Sections 3.5 and 3.7 currently overlap

Section 3.5's proposed taxonomy already routes through “optional
occupation/ground correction,” while Section 3.7 then introduces occupation
correction as its main physical story
(`design/chapter03_exact_source_contract.md:1680-1711`).

**Recommended resolution.** Section 3.5 should own the three partition-data
recipes and the synthesis-only low-temperature ground floor. It may name
density correction as a pending arrow but not explain it. Section 3.7 should
own Debye screening, ionization-potential lowering, and occupation gates
exactly once.

### P2. Apply Chapter 2's infrastructure silently and report only new evidence

The Chapter 3 verification ladder correctly requires table hashes, dtype,
shape, backend, and thread policy. Reader-facing prose should not repeat:

- the meaning of SHA-256 or the four data roles;
- the 25-field schema inventory or structural failure injection;
- definitions of tensor, dtype, device, thread, race, reduction, `njit`, or
  `prange`;
- cold/warm/cache timing methodology.

A concise manifest-bound table preflight, a real depth-`prange` source card,
and physics-specific parity results are sufficient. Generic infrastructure
belongs to Chapter 2 by the global duplication policy
(`design/global_chapter_contracts.md:656-664`).

### P2. Do not use the Chapter 2 schema fixture as the Chapter 3 thermodynamic state

Chapter 2 explicitly labels its four-depth archive a synthetic interface
fixture rather than a converged atmosphere
(`book/chapters/chapter_02.py:1494-1501`). Chapter 3 needs its own
manifest-bound atom-only integration fixture with declared \(T\), material
pressure, linear abundance, and seed electron density. The Chapter 2 fixture
may be used only for a continuity test of field names, not for a physical EOS
plot or parity claim.

### P2. The last Chapter 2 body paragraph briefly looks away from Chapter 3

The fixed-\(n_e\) paragraph gives the direct Chapter 3 dependency, but the next
paragraph returns to role metadata and Chapters 11–14 before the summary
(`book/chapters/chapter_02.py:1615-1630`). The summary repairs the handoff, so
this is not a blocker. On a later canonical polish pass, placing the metadata
paragraph before the full/fixed bridge would let the body itself end on the
question Chapter 3 immediately answers.

## Terminology, notation, axes, and units contract

Chapter 3 should use this table as a pre-draft fence.

| Quantity | Mathematical notation | Exact implementation name | Unit and shape |
| --- | --- | --- | --- |
| local temperature | \(T_d\) | `temperature` / `temperature_k` at exact call sites | K, `(D,)` |
| material gas pressure | \(P_{{\rm gas},d}\) | `gas_pressure` | dyn cm\(^{-2}\), `(D,)` |
| electron number density | \(n_{e,d}\) | `electron_density` | cm\(^{-3}\), `(D,)` |
| total nuclei number density | \(n_{{\rm nuclei},d}\) | `total_nuclei_number_density` | cm\(^{-3}\), `(D,)` |
| linear elemental number abundance | \(a_{d,s}\) | atmosphere `elemental_abundances_by_layer`; synthesis `elemental_abundances` | dimensionless relative number abundance, exact path-specific `(D,99)` or accepted source shape |
| actual population of stage \(r\) of element \(s\) | \(n_{s,r}\) | public `ion_stage_populations`; packed `ion_stage_populations_by_packed_slot` | cm\(^{-3}\) |
| partition function | \(U_{s,r}\) | path-specific partition result | dimensionless |
| stage population divided by partition | \(n_{s,r}/U_{s,r}\) | public `partition_normalized_populations`; packed `partition_normalized_populations_by_packed_slot` | cm\(^{-3}\) per dimensionless partition |
| bound-level population | \(n_i\) or \(n_{s,r,i}\) | no public Chapter 3 field | cm\(^{-3}\); derived from \((n/U)g_i e^{-E_i/kT}\) |
| atmosphere population layout | — | packed population fields | `(D,1006)`, sparse slots |
| synthesis internal EOS layout | — | `EOSResult` fields | `(D,99,6)`: depth, element, stored stage |
| public synthesis population layout | — | schema population fields | `(D,6,139)`: depth, stored stage, species |
| mass density | \(\rho_d\) | `mass_density` | g cm\(^{-3}\), `(D,)` |
| fractional Doppler width | \(\Delta v_D/c\) | `fractional_doppler_widths` | dimensionless; exact path layout |
| atomic specific internal energy | \(e_{\rm atom}\) | `specific_internal_energy` when stored | erg g\(^{-1}\), `(D,)` |

Additional fences:

- Depth index 0 remains outermost; increasing index moves inward. Atomic EOS
  depths are independent in Chapter 3 even though their storage order remains
  fixed.
- \(D\) means depth count everywhere. Do not use “layer” as a tensor axis name
  in one table and “depth” in another without stating the equality.
- Use “ion stage” for the physical charge state, “axis index” for zero-based
  storage, and “spectroscopic stage” for I, II, III labels.
- `partition_normalized_populations` is not an ion fraction, an actual stage
  population, or a bound-level population.
- Do not reuse Chapter 1's \(P_{\rm total}\) where the EOS requires
  `gas_pressure`.
- Do not imply the atmosphere and synthesis stacks share tables, constants,
  stage availability, mean-mass policy, stopping behavior, or pressure
  lowering merely because some outputs share names.

## Precise Chapter 3 opening contract

The first reader-facing movement should follow this exact causal order.

1. **One-sentence backward bridge.** “Chapter 1 supplied local
   `temperature` and `gas_pressure`; Chapter 2 supplied a declared linear
   mixture and named the still-empty population fields.”
2. **Concrete tension.** At one depth, compute only
   \(n_{\rm particle}=P_{\rm gas}/(k_{\rm B}T)\). Show two qualitatively
   different neutral/ion/electron allocations that fit the same total particle
   count. Do not present either as an equilibrium answer.
3. **Central question.**

   > Given \(T\), material gas pressure, and a linear elemental mixture, which
   > atomic levels and ion stages are occupied, and which \(n_e\) makes those
   > populations charge-consistent?

4. **Claim to earn.**

   > In an atom-only LTE gas, level weights and Boltzmann factors build each
   > partition function, Saha ratios connect ion stages, and a damped
   > charge-conservation fixed point closes the electron density.

5. **Assumptions before equations.** State atom-only ideal gas, LTE, local
   one-depth thermodynamics, inherited one-dimensional/static/plane-parallel
   storage, CGS units, outer-to-inner array order, and molecules disabled.
   Explain that molecule disabling is a chapter boundary, not a warm-star
   assertion.
6. **Two-path reads/writes table.** Use the exact atmosphere and synthesis rows
   specified in P1 above. The first production name may appear here because
   Chapter 2 already established these interfaces.
7. **Conceptual schematic.** Show
   levels + weights → \(U\) → Saha stage ladder ↔ charge closure. The feedback
   arrow belongs only between Saha populations and \(n_e\). Caption it as
   conceptual.
8. **First derivation and first numerical evidence.** Define level, excitation
   energy, and statistical weight; derive the two-level Boltzmann ratio; state
   the constant tier; predict the low/high-temperature behavior; execute the
   bite-size calculation; interpret the actual values. The first quantitative
   curve may now appear.
9. **Fulfil the Chapter 2 placeholder.** Build \(U\) from the two levels,
   compute \(n/U\), and verify \(n=U(n/U)\) using the exact public names only
   after the physical quantities have been related. Do not repeat the Chapter
   2 shape demonstration.

This opening uses one short backward recap and immediately advances the
physics. It contains no detached exercise and no result whose origin is
withheld.

## Required middle-chapter handoffs

The causal seams inside Chapter 3 should be explicit:

1. A two-level \(U\) cannot represent real atoms → three production partition
   recipes and stack-specific corrections.
2. Excitation within one stage cannot choose the ion stage → Saha.
3. Saha needs \(n_e\), while charge conservation determines \(n_e\) → damped
   fixed point.
4. One closed depth does not fill an atmosphere → real independent-depth
   `prange` and synthesis depth batching.
5. Particle counts alone do not supply downstream broadening/energy state →
   mass density, collision proxy, Doppler support, atomic internal energy.
6. Correct values are not yet in either engine's required representation →
   packed-to-public mapping.
7. A full closure and a fixed-\(n_e\) fill answer different questions → claim
   table and preservation/residual tests.

Each seam should have one owner. In particular, partition taxonomy should not
reappear inside the pressure-ionization section, and Chapter 2's generic
parallel/device explanations should not reappear inside the depth comparison.

## Precise Chapter 3 close and Chapter 4 handoff contract

The close must contain the following, in this order:

1. `## 3.N Chapter summary`.
2. Answer the opening question at the level actually computed: the atom-only
   LTE state is closed by Boltzmann/partition/Saha statistics plus charge
   conservation.
3. List only outputs demonstrated in the chapter:
   - atmosphere actual and partition-normalized `(D,1006)` packed arrays;
   - synthesis internal `(D,99,6)` and public `(D,6,139)` atomic arrays;
   - `electron_density`, `total_nuclei_number_density`, and `mass_density`;
   - fractional Doppler and atomic-energy support;
   - measured full-closure residual and exact fixed-\(n_e\) preservation.
4. State the exact claim distinction:
   - full closure changes \(n_e\) until the declared retained-stage charge
     residual passes its tolerance;
   - fixed-\(n_e\) preserves its supplied atmosphere density and does not
     establish charge closure.
5. State what is still unavailable. Chapter 3 has not built molecules,
   molecular internal energy, a full physically converged atmosphere,
   continuum or line opacity, radiative transfer, or a spectrum. It should not
   claim that its population subset alone is a complete schema-valid product.
6. Give the single causal missing dependency:

   > The atomic calculation allocates every nucleus to one element's ion
   > ladder. In cool layers, a molecule such as CO spends carbon and oxygen
   > together, so the element-by-element closure is no longer separable.

   CO may appear in the preceding body as the concrete preview; otherwise use
   “a molecule binding two elements” here so the summary introduces no new
   species.
7. `### Next: couple the elemental budgets`.
8. Link directly to
   `[Chapter 4: Molecules and Coupled Equilibrium](/reader.html?ch=4)` and say
   that Chapter 4 adds mass action, coupled conservation, positivity, and
   depth continuation. It reuses Chapter 3's atomic definitions and layouts
   but recomputes the coupled numerical atomic and electron state.

## Acceptance additions for the neighbor gate

In addition to the Chapter 3 exact-source checklist, the Chapter 1→2→3 gate
should fail if any of the following is present:

- an unexplained ion-fraction preview before Saha;
- a repeat of Chapter 2's \(10^{12}/4\) population example or schema inventory;
- a repeat of dex notation, checksum failure injection, Numba syntax, device
  discovery, or cache timing;
- `total_pressure` supplied to an EOS boundary;
- a logarithmic abundance array supplied where the EOS contract says linear;
- an ambiguous “stage 1” without charge/spectroscopic/axis context;
- a silent switch from Chapter 1 exact constants to EOS reference constants;
- a fixed-\(n_e\) route credited with closure or fed an arbitrary density while
  being described as an accepted atmosphere handoff;
- a claim that `(D,99,6)` and `(D,6,139)` are the same layout;
- a claim that the Chapter 2 schema fixture is a physical atmosphere;
- molecular population work, mass-action derivation, or depth continuation
  taught before Chapter 4;
- a summary that introduces a new molecule, API, field, or numerical result.

With these conditions enforced, Chapter 2's two-movement density decision is
coherent: Chapter 3 receives a complete numerical and data vocabulary, spends
no opening time rebuilding it, and can devote its full narrative budget to
the first real physical state.
