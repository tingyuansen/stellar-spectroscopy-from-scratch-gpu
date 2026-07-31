# Chapter 4 → 5 → 6 neighbor audit

Status: central pre-draft flow lock  
Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`  
Audited on: 2026-07-29

This audit checks the local causal seam around the continuum chapter. It is a
reader-flow contract, not a second Chapter 5 outline. The exact scientific
authority remains `design/chapter05_exact_source_contract.md`.

## 1. The three questions form one causal chain

| chapter | question now answered | concrete output handed forward |
| --- | --- | --- |
| 4 — Molecular equilibrium | Which particles exist at each depth after atomic and molecular conservation close together? | a packed atmosphere/runtime consumer view and a separate validated schema-v4 synthesis view |
| 5 — Continuous opacity and scattering | How strongly do those particles absorb or redirect photons away from line centres? | separate smooth absorption and scattering slabs, an atmosphere thermal-source column, and the atmosphere line-selection threshold |
| 6 — One spectral line | How does one bound-bound transition acquire an integrated strength, width, damping, and normalized profile above that background? | one checked line-opacity column that can later be selected and accumulated in bulk |

The transition is causal rather than chronological: Chapter 4 supplies
particles, Chapter 5 lets them interact continuously with light, and Chapter 6
adds the first wavelength-localized transition.

## 2. Chapter 4 closes exactly where Chapter 5 begins

### What Chapter 5 may consume without rederiving

- `temperature`, `mass_density`, and `electron_density`;
- named actual neutral and ionic densities;
- hydrogen-stage and element-stage partition-normalized populations;
- the actual `ion_stage_populations` cube for charge-square free-free terms;
- atmosphere-only packed, partition-normalized CH and OH slots;
- hydrogen departure-coefficient state where the atmosphere continuum owns it;
- schema edge geometry for the synthesis lane.

The first mention of each field in reader prose still follows its physical
meaning. The complete implementation-name table appears only after the process
budget and grid algorithms are understood.

### What Chapter 5 must not reopen

- Saha ionization balance;
- partition-function construction;
- the 170-record atmosphere molecular solve;
- the 190-record synthesis molecular solve;
- electron-density closure or depth continuation;
- schema-v4 construction.

One local definition,
\(\widetilde n_{i,r}=n_{i,r}/U_{i,r}\), is enough to explain why a bound-level
sum reads a normalized base. It is not permission to repeat Chapter 3.

### The H2 seam is intentionally asymmetric

Chapter 4 hands forward two distinct closed state views. The atmosphere
continuum consumes the packed `ModelAtmosphere`/`RuntimeState` adapter,
including departure coefficients and 1006-slot population arrays. Synthesis
consumes schema v4. The atmosphere packed state must never be reconstructed
from schema v4.

It also hands forward several legitimate H2 representations, but neither
continuum product consumes the stored schema
`molecular_hydrogen_population`:

1. the atmosphere continuum reconstructs H2 locally with its 200-point
   partition table;
2. the synthesis continuum reconstructs H2 locally with its analytic policy;
3. CH and OH use partition-normalized packed slots 845 and 847; their
   cross-section helpers restore the partition factors;
4. H2-plus absorption is a separate ionic process and must never be called
   H2 collision-induced absorption.

The reader sees this distinction once, in Chapter 5's molecular-continuum
section. Later chapters may use the stored H2 field for their own declared
consumer, but must not imply that it drove the continuum.

### The edge-array seam has one consumer

The schema carries:

- `signed_continuum_edge_frequency_hz`;
- `continuum_edge_wavelength_nm`;
- `continuum_edge_midpoint_wavelength_nm`;
- `continuum_edge_interval_width_squared_over_two_nm2`.

These arrays serve standard synthesis continuum interpolation. The atmosphere
continuum does not consume them: it independently builds and directly
evaluates an effective-temperature-dependent 30,000-point grid. Chapter 5
must show these as two exact algorithms, not two backends of one invented grid.

## 3. Chapter 5 has one physical spine and distinct products

The shared causal identity is

\[
\kappa_\nu=\frac{n_{\rm absorber}\sigma_\nu}{\rho}.
\]

It is used once to teach how microscopic interaction area becomes opacity per
gram. From there Chapter 5 keeps four execution lanes distinct:

| lane | exact destination | must not be claimed |
| --- | --- | --- |
| atmosphere product | direct 30,000-point CPU/Numba float64 absorption, scattering, and absorption-weighted source | synthesis interpolation, Torch execution, or stored-schema H2 consumption |
| synthesis product | used edge triplets, log-parabolic interpolation, device tensors, Coulomb layout `False`, no `FrequencyInvariants` | atmosphere CH/OH/CIA or full direct-grid evaluation |
| sampled diagnostic | caller frequencies, Coulomb layout `True`, Planck-source diagnostic | the standard public synthesis route |
| sampled extension | explicit `FrequencyInvariants` and its extended helper set | a performance-only alias or unconditional whole-domain equality |

Shared physics is explained once. Lane-specific constants, tables, sum order,
floors, state adapters, and dtype policies are exposed only at the point where
they change the executable result.

## 4. Exact artifact passed from Chapter 5 to Chapter 6

Chapter 5 closes with:

- standard synthesis `continuum_absorption (D,W)`;
- standard synthesis `continuum_scattering (D,W)`;
- derived synthesis
  `continuum_opacity = continuum_absorption + continuum_scattering`, the
  extinction scale consumed by line selection rather than a third physical
  component;
- atmosphere direct-grid absorption, scattering, and source `(D,30000)`;
- atmosphere line-selection threshold `(D,344)` float32, including the exact
  duplicate last column and sentinel policy;
- named component budgets retained for verification;
- no bound-bound opacity.

Chapter 6 reads the synthesis continuum background when it demonstrates the
line-strength cutoff and one-line opacity. It does not need the atmosphere
30,000-point grid or the atmosphere thermal-source array. Those remain
atmosphere-consumer products for the later atmosphere chapters.

## 5. Chapter 6 starts with the first genuinely new physics

The first new object is a bound-bound transition. Chapter 6 owns:

- lower-level population for one line;
- oscillator strength;
- integrated line strength and stimulated emission in the line convention;
- thermal and microturbulent Doppler width;
- natural, Stark, and van der Waals damping;
- normalized Voigt/Harris/FASTEX profile;
- one-depth and all-depth line opacity.

Chapter 5 may use the phrase “narrow line forest” to motivate what is missing,
and may construct the continuum threshold later consumed by selection. It
must not teach oscillator strengths, line damping, Voigt profiles, wing
reach, catalog masks, or sparse line accumulation.

Chapter 6 may recall that the smooth background is
`continuum_absorption + continuum_scattering` for a selection comparison. It
must not rederive any continuum cross section, Rayleigh law, edge triplet,
interpolation basis, IFOP group, or Numba continuum loop.

## 6. Forward- and backward-reference budget

### Chapter 4 closing paragraph

One backward-free handoff is sufficient:

> The particle inventory is closed. Chapter 5 now asks how those particles
> absorb or redirect photons without reopening the equilibrium solve.

The current Chapter 4 closing satisfies this contract.

### Chapter 5 internal references

- one short recap of actual versus normalized population roles;
- one explicit statement that molecular equilibrium is supplied;
- one honest forward boundary when the line-reference threshold is built:
  Chapter 6 will explain the comparison, while this chapter owns the continuum
  quantity;
- no “as we will see repeatedly” promises.

### Chapter 5 closing paragraph

One causal link is sufficient:

> The smooth background is now accounted for, yet the opening spectrum still
> contains narrow features that no continuum process can make. Chapter 6 gives
> one bound-bound transition a strength, width, and profile.

### Chapter 6 opening

Use the checked continuum as a supplied local background. Do not summarize
Chapter 5's process garden.

## 7. Redundancy locks

| idea | sole teaching owner | later use |
| --- | --- | --- |
| actual vs partition-normalized population | Ch. 3, applied once in Ch. 5 | named inputs only |
| molecular mass action and closure | Ch. 4 | supplied CH/OH and consumer-specific H2 policy |
| cross section to mass opacity | Ch. 5 | invoked, never rederived |
| continuum stimulated-emission factor | Ch. 5 | line convention distinguished briefly in Ch. 6 |
| bound-free/free-free/CIA | Ch. 5 | continuum slab only |
| Thomson/Rayleigh scattering | Ch. 5 | transfer consumes scattering fraction later |
| atmosphere direct grid and `prange` | Ch. 5 | atmosphere runner composes it later |
| synthesis edge triplets and interpolation | Ch. 5 | standard synthesis composes it later |
| one bound-bound line | Ch. 6 | catalog chapters route and accumulate it |

No section may present a source-file tour as a substitute for this ownership
map. Exact names are discoverable in compact implementation contracts and
live cells, while the prose remains organized by physical questions.

## 8. Acceptance checks for the Chapter 5 draft

The draft is locally coherent only if all answers are “yes.”

- Does the opening observation require a continuum explanation without
  presupposing process names?
- Does Chapter 5 consume the Chapter 4 state without running chemistry?
- Are stored H2, atmosphere-local H2, and synthesis-local H2 visibly distinct?
- Is atmosphere direct sampling independent of the schema edge arrays?
- Are atmosphere and synthesis products both complete but never equated?
- Are absorption, scattering, and the thermal numerator kept separate?
- Is every useful limit or perturbation part of the main text rather than an
  exercise section?
- Does the line-reference threshold stop before teaching line selection?
- Does the summary state what was built, rather than list routines?
- Does the final sentence make one bound-bound line the inevitable next
  question?

The same audit is rerun after the executable notebook exists, using its actual
section order, visible cells, plots, and closing link rather than this design
intent.
