# Chapter 3 Causal Outline — Atoms, Ions, and Electrons

This is the novice-first narrative contract for canonical Chapter 3. It is subordinate to
`BIBLE.md`, `design/pedagogical_flow_rubric.md`,
`design/global_chapter_contracts.md`, and
`design/chapter03_exact_source_contract.md`. It does not authorize a second EOS
implementation, a reader dependency on the pinned Payne Zero checkout, or a source dump in
Markdown.

The exact source contract inventories fifteen necessary teaching movements. This outline combines
them into thirteen numbered sections and three internal lesson movements. The combination is
deliberate: the reader should experience one causal argument, not fifteen unrelated API tours.

## Central question and earned claim

**Question.** Given temperature, gas pressure, and composition at one depth, how many particles
occupy each excitation and ionization state, and what electron density makes that division
consistent with charge conservation?

**Claim the reader must earn.**

> Temperature and density set relative atomic populations, but the absolute state exists only
> after particle conservation and charge conservation agree on the same electron density.

This is the chapter's durable spine:

```text
levels and statistical weights
    → partition function
    → adjacent ion-stage ratios
    ↔ electron-density charge closure
    → densities, Doppler support, and atomic energy
    → exact packed and synthesis layouts
```

The feedback arrow is the conceptual center. Everything before it determines the charge implied
by a trial `electron_density`; everything after it is trustworthy only once that trial has been
accepted or explicitly preserved as a fixed-\(n_e\) input.

## Neighbor handoff and no-repeat contract

Chapter 2 ends by naming `electron_density`, `ion_stage_populations`, and
`partition_normalized_populations` while explicitly saying that none of their values has yet been
earned. Chapter 3 begins from that exact tension. Its first paragraph should echo the handoff once:

> The schema has places to store electrons and ion populations, but an empty field name does not
> tell us how many particles belong there.

Chapter 3 reads, without rederivation:

| object | shape | unit | dtype/device | inherited meaning |
| --- | --- | --- | --- | --- |
| `temperature` | `(D,)` | K | NumPy `float64`, CPU at the atmosphere boundary | local thermal state |
| `gas_pressure` | `(D,)` | dyn cm\(^{-2}\) | NumPy `float64`, CPU | total material pressure |
| `elemental_abundances` or `elemental_abundances_by_layer` | `(99,)` or `(D,99)` | linear relative number abundance | NumPy `float64`, CPU | composition, already distinguished from dex notation in Chapter 2 |
| supplied `electron_density` where the fixed bridge is tested | `(D,)` | cm\(^{-3}\) | NumPy `float64`, CPU | an upstream value to preserve, not a value silently solved here |

It writes:

- actual ion-stage populations;
- ion-stage populations divided by partition functions;
- a full atom-only electron-density closure and its residual;
- `total_nuclei_number_density` and `mass_density`;
- the atom-only neutral-collision proxy and `fractional_doppler_widths`;
- atomic `specific_internal_energy`;
- exact atmosphere `(D,1006)`, internal synthesis `(D,99,6)`, and public synthesis
  `(D,6,139)` representations;
- a claim table that distinguishes full charge closure from a fixed-\(n_e\) population fill.

The chapter must not re-teach:

- array axes, broadcasting, strides, or the meaning of dtype/device;
- what `njit`, `parallel=True`, `prange`, Torch, CUDA, or MPS are;
- checksum mechanics, the four data roles, or the complete schema-v4 inventory;
- abundance dex conversions or the Chapter 2 direct-abundance examples;
- the Planck function, optical-depth integration, or radiative transfer;
- molecular mass action, opacity, damping, or line profiles.

Instead, it applies Chapter 2's contracts. A compact table may say that atmosphere closure is CPU
NumPy/Numba `float64`, while synthesis population algebra is Torch on the selected device and work
dtype. It should not repeat the device-selection tutorial.

## How the fifteen source-contract movements become one chapter

| exact-contract movement | reader-facing home here | reason for combination |
| --- | --- | --- |
| 3.1 The missing electrons | 3.1 | opening physical tension |
| 3.2 The chapter state contract | 3.1 | assumptions belong beside the opening claim |
| 3.3 Why one energy has several states | 3.2 | first physical construction |
| 3.4 The partition function is a weighted inventory | 3.3 | normalization follows the two-level ratio immediately |
| 3.5 Real atoms need more than one partition recipe | 3.4 | first production convergence |
| 3.6 Removing an electron | 3.5 | Saha opens the ion ladder |
| 3.7 Dense plasma lowers the last bound levels | 3.6 | correction to the just-built ladder |
| 3.8 One element exposes electron feedback | 3.7 | smallest charge fixed point |
| 3.9 Ninety-nine elements close one depth | 3.8 | production closure |
| 3.10 Depths are independent here | 3.8 | acceleration is the same physical closure over independent rows |
| 3.11 Density and broadening support | 3.9 | downstream support bundle |
| 3.12 Where ionization energy is stored | 3.10 | final thermodynamic output of the closed atomic state |
| 3.13 Two exact layouts | 3.11 | interface movement |
| 3.14 Full closure or fixed-\(n_e\) | 3.12 | final physical-claim boundary |
| 3.15 Summary and handoff | 3.13 | close |

The chapter uses three visible movement dividers:

1. **Movement I — What temperature and density do to one atom**
2. **Movement II — Let charge close the material state**
3. **Movement III — Move the state without changing its claim**

These are pacing anchors, not extra navigation chapters.

## Terminology and notation ladder

New terms must appear in this order:

1. **level** — one allowed energy within a fixed ion stage;
2. **statistical weight** \(g_i\) — the number of quantum states represented by a listed level;
3. **Boltzmann factor** — the relative thermal preference for a level;
4. **partition function** \(U_{s,r}\) — the weighted inventory that normalizes levels within one
   ion stage;
5. **ion stage** — the same nucleus after losing a stated number of electrons;
6. **ionization energy** \(\mathcal I_{s,r}\) — the energy needed to move to the next stage;
7. **Saha ratio** — the equilibrium ratio between neighboring stages at a stated \(T\) and \(n_e\);
8. **charge residual** — electron density minus the positive charge implied by the stage
   populations;
9. **fixed point** — a value that is unchanged when the population-and-charge update is applied;
10. **pressure ionization / occupation correction** — the dense-plasma modification that dissolves
    the highest bound states;
11. **perturber proxy** and **fractional Doppler width** — support quantities, introduced only
    after actual populations exist.

The notation is fixed throughout:

| symbol | meaning | unit |
| --- | --- | --- |
| \(s\) | element label | none |
| \(r\) | mathematical ion stage, with neutral as the first stage | none |
| \(i\) | bound level within one stage | none |
| \(E_i\) | excitation energy above that stage's ground level | erg, eV, or cm\(^{-1}\) only when the conversion is explicit |
| \(g_i\) | statistical weight | dimensionless |
| \(n_i\) | actual bound-level number density | cm\(^{-3}\) |
| \(n_{s,r}\) | actual number density of one ion stage | cm\(^{-3}\) |
| \(U_{s,r}\) | partition function | dimensionless |
| \(n_e\) | electron number density | cm\(^{-3}\) |
| \(n_{\rm nuclei}\) | total nuclei number density | cm\(^{-3}\) |
| \(q_r\) | positive ionic charge in units of the elementary charge | dimensionless |

At the earned implementation boundary, state explicitly:

```text
n_{s,r}                 → ion_stage_populations
n_{s,r} / U_{s,r}       → partition_normalized_populations
Δv_D / c                 → fractional_doppler_widths
```

`partition_normalized_populations` must never be called a fraction, a bound-level population, or
an actual ion population. The identity

\[
n_i =
\left(\frac{n_{s,r}}{U_{s,r}}\right)
g_i e^{-E_i/(k_{\rm B}T)}
\]

is shown once to establish meaning. Chapter 6 later uses it in line opacity; Chapter 3 does not
continue into an opacity formula.

## Movement I — What temperature and density do to one atom

### 3.1 Same mixture, different electrons

**Inherited question.** Chapter 2 can validate an `electron_density` array, but what determines its
values?

Open with two imagined layers containing the same elemental mixture and the same stated gas
pressure, one cool and one hot. Ask the reader to picture hydrogen in both:

- in the cool layer, thermal collisions rarely supply the ionization energy;
- in the hot layer, more collisions can pay the ionization cost;
- every released electron changes the density that enters the next ionization calculation.

Do not show a precomputed population plot yet. The first numerical population should be one the
reader has derived.

State assumptions before equations:

- local thermodynamic equilibrium;
- ideal-gas, atom-only material;
- one-dimensional, static atmosphere inherited from Chapter 1;
- no molecules in this chapter's closure;
- no opacity or radiation field in the EOS calculation;
- depth index runs outermost to innermost, although each atom-only depth is independent here;
- atmosphere work uses CPU NumPy/Numba `float64`;
- synthesis uses its declared Torch device/dtype policy and retains discrete interpolation choices
  on host `float64`.

Give one compact reads/writes contract:

```text
reads:  temperature (D,), gas_pressure (D,), composition (D,99)
writes: electron and nuclei densities (D,),
        actual and partition-normalized populations,
        mass/Doppler/energy support
```

Do not list all `PopulationState` fields here. Exact names arrive when their physical objects have
been constructed.

**Opening claim.**

> Temperature can redistribute particles, but charge conservation decides whether that
> redistribution is self-consistent.

**Exact transition to 3.2.**

> Before an atom can lose an electron, its remaining electrons can already occupy different
> energies. We therefore begin inside one ion stage, where charge is not yet changing.

### 3.2 Excitation: one ion stage contains several levels

Introduce a level with a two-rung ladder. The ground level has energy zero; the upper level has
excitation energy \(\Delta E\). Introduce \(g=2J+1\) only as a concrete example of statistical
weight: it counts how many states share the listed energy. Do not teach angular-momentum coupling.

Derive in short steps:

\[
\frac{n_1}{n_0}
=
\frac{g_1}{g_0}
\exp\!\left[-\frac{E_1-E_0}{k_{\rm B}T}\right].
\]

For the synthetic teaching atom use exactly:

```text
energy_cm = [0, 10000]
statistical_weight = [2, 4]
temperature checkpoints = [3000, 6000, 12000] K
```

Explain the wavenumber conversion \(E=hc\tilde\nu\) before code. This is a controlled two-level
model, not a production atom and not a static source table.

**Prediction before code.**

- At low \(T\), the upper level is rare.
- The ratio increases monotonically with \(T\).
- As \(T\rightarrow\infty\), the ratio approaches \(g_1/g_0=2\), not infinity.

**Visible cell C1 — two-level Boltzmann ratio and one-panel figure.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| `temperature_grid (N,)` K, `energy_cm (2,)`, `statistical_weight (2,)` | `excited_to_ground_ratio (N,)` and three checkpoint values | NumPy `float64`, CPU |

The cell should be 15–25 lines and make the one-panel plot itself. Use a dark reference curve,
three muted-blue checkpoint markers, logarithmic \(y\) only if the actual range needs it, and axis
labels `Temperature (K)` and `Excited / ground population`. Draw the high-\(T\) limit as a quiet
grey guide. There is no legend if direct labels remain clearer.

**Required output interpretation.** Read the actual ratios at 3000, 6000, and 12000 K. State that
they increase and remain below two. Say explicitly that the result is a ratio inside one ion
stage; it does not yet tell us what fraction of the element is neutral or ionized.

**Exact transition to 3.3.**

> A ratio compares two levels, but a population calculation needs fractions whose sum is one. The
> missing normalizer is the partition function.

### 3.3 The partition function is a weighted inventory

Define

\[
U_{s,r}(T)=\sum_i g_i e^{-E_i/(k_{\rm B}T)}
\]

as the weighted inventory of levels available to element \(s\), ion stage \(r\). Each term has a
plain-language interpretation: number of states times thermal accessibility.

Then derive

\[
\frac{n_i}{n_{s,r}}
=
\frac{g_i e^{-E_i/(k_{\rm B}T)}}{U_{s,r}},
\qquad
\sum_i\frac{n_i}{n_{s,r}}=1.
\]

**Prediction before code.** The ground and excited fractions should add to one at every
temperature. Increasing \(T\) should move population upward without creating particles.

**Visible cell C2 — normalize the two-level atom.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| the C1 two-level arrays and a declared `ion_stage_population` | `partition_function`, two level fractions, `ion_stage_population / partition_function`, reconstructed level populations | NumPy `float64`, CPU |

Print one compact table with temperature, \(U\), both level fractions, and their sum. Print the
maximum reconstruction error for

\[
n_i=(n_{s,r}/U_{s,r})g_i e^{-E_i/(kT)}.
\]

**Required output interpretation.** Read the maximum sum-to-one and reconstruction errors rather
than saying only “the test passes.” Explain why a later line calculation can reuse \(n_{s,r}/U\)
for many levels.

Only now introduce the exact names:

- `ion_stage_populations`: actual \(n_{s,r}\), cm\(^{-3}\);
- `partition_normalized_populations`: \(n_{s,r}/U_{s,r}\),
  cm\(^{-3}\) per partition function.

**Exact transition to 3.4.**

> The two-level sum is transparent because we supplied every level. Real atoms need several
> source-native ways to represent the same weighted inventory, and those ways are not
> interchangeable.

### 3.4 Production partition functions: three recipes, two policies

Open with the physical need, not a table inventory: real atoms have too many levels, incomplete
high-lying series, and density-sensitive bound states. One universal smooth classroom function
would conceal real source decisions.

Introduce the production taxonomy in this order:

```text
explicit ordered special-level sum
or packed ordinary-ion interpolation
or PFIRON interpolation for Z = 20…28
→ stack-specific low-temperature or occupation policy
→ U
```

The exact atmosphere and synthesis table stacks stay separate. Explain only the differences that
will affect the chapter's computations:

- packed tables and ionization-potential tables are not identical;
- synthesis has additional helper stages and a synthesis-only low-temperature ground floor;
- PFIRON uses a `(7,56,10,9)` lowering/temperature/stage/element grid;
- both stacks may extrapolate in temperature;
- above the final lowering node, atmosphere clamps while synthesis extrapolates from its final two
  planes.
- although the synthesis bundle includes Ca I and Ca II special-level arrays, its normal
  element-dispatch route sends \(Z=20\ldots28\) to PFIRON before those special cases.

Do not list hundreds of branch labels or terms.

**Visible cell C3 — verify the Chapter 3 data inventory before using it.**

This is the only checksum cell. It applies, rather than re-teaches, Chapter 2's data policy.

Contract:

| reads | writes | execution |
| --- | --- | --- |
| Chapter 3 manifest; separate atmosphere and synthesis static table archives; role-separated one-element fixture | compact table of role, source-stack label, array count, byte hash status | Python/NumPy, CPU |

The output must prove that:

- atmosphere and synthesis static tables have separate manifest identities;
- the mixed source synthesis bundle has been split into static tables and a depth-specific
  integration fixture;
- `ground_partition_table (605,80)` is recorded as a loaded fixture field, not described as the
  active runtime ground lookup;
- hashes, shapes, dtypes, units, and builders are present.

**Required output interpretation.** State which distinct assets were verified. Repeat the Chapter
2 caveat in one clause only: matching bytes establish identity, not physical correctness.

At the first exact calculation, identify the rounded reference constants used by the EOS source,
including \(k_{\rm B}\), \(h\), the atomic mass unit, wavenumber per eV, and the one-spin Saha
coefficient. Do not replace them with a universal constants module or newer CODATA values.

**Visible cell C4 — probe the exact partition branches.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| manifest-bound special, packed ordinary, and PFIRON tables; the four-temperature exact branch fixture | partition values for H, one ordinary element, and Fe; stack/policy labels; ground-floor and PFIRON-edge diagnostics | exact local progressive-package functions; CPU NumPy `float64` plus declared Torch work dtype/device where synthesis is evaluated |

This cell may call a clearly labeled verification helper, but that helper must call the canonical
exact functions and must not become a new public EOS API. Keep the output to one compact table.

Predictions:

- every reported production partition is at least one;
- different branch types need not agree numerically;
- the synthesis ground correction acts as a floor, not an added term;
- at a lowering coordinate above 32000 cm\(^{-1}\), atmosphere and synthesis need not agree because
  their edge policies differ.

**Required output interpretation.** Name the actual H/ordinary/Fe cases and read the measured
policy difference. Do not call either stack an approximation to the other. State that these are
separately certified stages.

One short exact-source card is allowed here only if it is at most 15 lines and demonstrates the
ordered nature of `ground_partition_value`. It must be followed by an explanation of why order and
the max-floor policy matter. Do not paste packed interpolation or special dispatches.

**Exact transition to 3.5.**

> The partition function distributes particles among levels only after an ion stage is chosen.
> We still need the rule that chooses how much of an element is neutral, singly ionized, or more
> highly ionized.

### 3.5 Removing an electron: the Saha ladder

Begin with the reaction in words:

```text
ion stage r ↔ ion stage r+1 + one free electron
```

Then build the adjacent-stage equation one physical factor at a time:

\[
\frac{n_{s,r+1}n_e}{n_{s,r}}
=
\frac{2U_{s,r+1}}{U_{s,r}}
\left(\frac{2\pi m_e k_{\rm B}T}{h^2}\right)^{3/2}
\exp\!\left[
-\frac{\mathcal I_{s,r}-\Delta\mathcal I_{s,r}}{k_{\rm B}T}
\right].
\]

Explain:

- the partition ratio counts internal states on the two sides;
- the \(T^{3/2}\) factor counts the free electron's translational states;
- the exponential pays the ionization-energy cost;
- multiplying by \(n_e\) means electron crowding favors recombination.

Define every symbol and unit. Include one dimensional check showing that the right side has number
density units. Treat \(\Delta\mathcal I=0\) for the first transparent calculation; Section 3.6 adds
the dense-plasma correction.

Place the first original schematic here, after level, partition, ion stage, and electron density
have meanings. Read it from left to right, then stop at the feedback arrow and say that Sections
3.7–3.8 will close that unresolved loop. Its caption must call it conceptual.

**Schematic S1 — level ladder to charge closure.**

- Landscape, original textbook composition.
- Left: three short energy rungs with unequal statistical-weight dots.
- Middle-left: a box labeled `partition sum U`.
- Middle-right: neutral, +1, +2 ion cards connected upward.
- Right: `electron density n_e` and `charge sum` connected by one feedback arrow.
- The only notation contrast should be `actual n` versus `n/U`.
- No atoms-as-planets artwork, decorative plasma, numerical values, or copied website layout.
- Alt text should give the left-to-right path and identify the feedback arrow.

Predictions before code:

- lower \(T\), larger ionization energy, or larger \(n_e\) favors the lower stage;
- higher \(T\) or smaller \(n_e\) favors the upper stage;
- the complete internal ladder normalizes to one;
- a returned subset may sum to less than one when helper stages are not stored.

**Visible cell C5 — H I/H II versus temperature.**

Use a readable log-space teaching helper for the adjacent-stage ladder. Label it pedagogical, not
the exact atmosphere algorithm.

Contract:

| reads | writes | execution |
| --- | --- | --- |
| declared fixed `electron_density`, hydrogen ionization energy, exact branch partitions over a temperature grid | H I and H II fractions, truncation deficit | NumPy `float64`, CPU |

Make one panel with temperature in K on the horizontal axis and ion fraction on the vertical axis.
Use dark navy for H I and muted orange for H II. Direct-label curves if possible. Mark the crossing
temperature found from the computed arrays. The figure makes one claim: temperature changes the
dominant ion stage at fixed \(n_e\).

**Required output interpretation.** Read the actual cool-end and hot-end fractions and the
crossing temperature. State that the curves are conditional on the declared fixed electron
density, so the plot is not yet a closed gas state.

Now converge to exact implementations:

- atmosphere `saha_partition_depth` forms direct `float64` ratios and a reverse normalization;
- synthesis uses a log-space cumulative ladder on the selected Torch device/dtype;
- the atmosphere scalar kernel clamps temperature to at least 1 K and electron density to at least
  `1e-40` cm\(^{-3}\); extreme teaching values use the stable helper first, while exact parity is
  claimed only on the declared physical fixture;
- atmosphere mode 11 returns stage fraction divided by \(U\);
- mode 12 returns actual stage fraction;
- mode 13 returns \(U\);
- the scalar atmosphere function does not itself apply abundance or nuclei-density scaling even
  though two descriptive parameters appear in its signature.

**Visible cell C6 — exact modes and representation identities.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| exact one-element fixture for H, one ordinary element, and Fe; separate source tables | mode-11, mode-12, mode-13 arrays; stored-subset sum; truncation deficit; reconstruction error | canonical atmosphere scalar and batch paths, NumPy `float64`, CPU |

First compute all outputs locally. Then, and only then, load
`chapter03_atmosphere_saha_outputs.npz` and compare table/fixture/source hashes before comparing
arrays.

The output must report:

- scalar row versus batch row;
- \(f_r=U_r(f_r/U_r)\) where \(U_r>0\);
- nonnegative returned fractions;
- returned sum at most one;
- explicit truncation deficit rather than a false conservation failure.

**Required output interpretation.** Name the actual maximum differences and whether equality is
exact in the pinned CPU environment. Make clear that mode outputs become number densities only
after the caller multiplies by nuclei density and the element's linear abundance.

**Exact transition to 3.6.**

> The isolated-atom Saha ladder assumes each bound level remains distinct. In a dense plasma,
> neighboring charges screen the nucleus and the highest levels can dissolve, changing both the
> ionization cost and the partition inventory.

### 3.6 Dense plasma changes the last bound levels

Introduce the Debye radius as the distance over which surrounding charges screen an electric
field:

\[
r_D=
\sqrt{\frac{k_{\rm B}T}{4\pi e^2 n_{q^2}}}.
\]

Explain the physical direction before source details:

- larger charge density shortens \(r_D\);
- shorter \(r_D\) lowers the energy required to ionize;
- the highest bound levels cease to behave as isolated states.

Then show

\[
\Delta\mathcal I_1
=
\min\!\left(1,\frac{1.44\times10^{-7}}{r_D}\right)\ {\rm eV},
\]

followed by the source stage-charge multiplier. Explain that the one-eV cap acts before that
multiplier.

The exact-boundary box must distinguish, concisely:

- atmosphere explicit `charge_square_density` versus the synthesis pressure/electron proxy;
- rounded source denominator literals versus the idealized \(4\pi e^2\) expression;
- special-branch versus ordinary-branch occupation gates;
- synthesis-only low-temperature ground correction;
- PFIRON lowering-edge clamp versus extrapolation.

Give the atmosphere proxy explicitly because C7 consumes it:

```text
q2 = 2 * electron_density
excess = 2 * electron_density - gas_pressure / thermal_energy_erg
if excess > 0:
    q2 += 2 * excess
```

During atom-only atmosphere closure this proxy is replaced by
\(n_e+\sum_{s,r}q_r^2n_{s,r}\). The synthesis stack continues to use its pressure/electron proxy
on each evaluation.

Use one small policy table rather than several paragraphs:

| exact branch | correction gate that matters here |
| --- | --- |
| atmosphere special | positive returned density parameter; no ordinary \(2T_{\rm ref}\), \(4T_{\rm ref}\), or 0.1-eV gate sequence |
| atmosphere ordinary | uncorrected below \(2T_{\rm ref}\); later requires a valid weight, stage lowering at least 0.1 eV, and \(T\ge4T_{\rm ref}\) |
| synthesis special | low-temperature block first; positive special occupation parameter bypasses the later high-\(T\)/0.1-eV gates |
| synthesis ordinary | low-temperature block first, then the high-\(T\), lowering, and weight gates |

Record one further special-level difference without expanding it into a detour: synthesis gates
explicit H I, He I, and He II excited-level terms below 9000, 15000, and 30000 K respectively,
whereas the atmosphere special kernel does not use those same temperature gates. C7 may use H to
make one of these policies visible. This box reports policy; it does not ask the reader to memorize
every threshold.

**Visible cell C7 — one controlled lowering and occupation gate crossing.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| exact H and Fe branch fixture with declared `temperature`, `gas_pressure`, `electron_density`, and `charge_square_density` | Debye radius, per-charge lowering, effective ionization energy, before/after partition or stage ratio, branch-policy label | exact atmosphere and synthesis stack probes under their declared dtype/device policies |

Predictions:

- increasing charge density cannot lengthen the Debye radius;
- the per-charge lowering cannot exceed 1 eV;
- crossing an occupation gate can change a partition without making it less than one;
- the two stacks are not expected to agree across intentionally different policy edges.

**Required output interpretation.** Read the actual gate status and values before and after the
crossing. Do not imply that this one fixture validates all special branches.

**Honest boundary.** We can now compute stage populations for a supplied \(n_e\). We still cannot
claim the gas is charge-neutral because the populations themselves determine how many electrons
should exist.

**Movement-I bridge.**

> A trial electron density chooses an ionization balance, and that ionization balance predicts a
> new electron density. Movement II closes this loop instead of treating either value as an
> independent input.

## Movement II — Let charge close the material state

### 3.7 One element exposes the electron feedback

Start from the ideal-gas particle count:

\[
n_{\rm particle}
=
\frac{P_{\rm gas}}{k_{\rm B}T}
=
n_{\rm nuclei}+n_e.
\]

Then write charge conservation:

\[
n_e=\sum_s\sum_r q_r n_{s,r}.
\]

Explain why these equations form a feedback loop:

1. guess \(n_e\);
2. obtain \(n_{\rm nuclei}=P/(kT)-n_e\);
3. use Saha to divide an element among stages;
4. sum its positive charge;
5. update \(n_e\);
6. repeat until applying the update changes nothing.

Define a fixed point here in plain language. Say explicitly that neither production stack uses a
Newton method for atom-only electron closure; no derivative or Jacobian is formed.

Introduce the exact damped update:

```text
raw      = charge implied by current populations
bounded  = max(raw, 0.5 * old)
new      = 0.5 * (bounded + old)
residual = abs((old - new) / max(new, 1e-300))
```

Interpret each line. From the lower bound and half-step, predict that one update cannot reduce the
old density below 75% of its previous value.

**Visible cell C8 — trace a one-element fixed point.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| one temperature, pressure, one-element abundance, positive electron seed, exact Saha branch | electron-density trace, raw charge trace, relative residual | readable scalar Python/NumPy `float64`, CPU, same arithmetic order as the extracted update |

Make one semilog panel of residual versus iteration. Mark the declared tolerance with a quiet grey
line. The code must print the first three updates, final \(n_e\), final residual, and number of
iterations.

**Required output interpretation.** Point to the initially damped movement and the eventual
residual crossing. Compare the observed largest one-step decrease with the predicted 25% bound.
State that this one-element trace explains the mechanism but is not the complete mixture.

Include one inline precondition check: a seed producing nonpositive
\(P/(kT)-n_e\) is not rescued by the atmosphere scalar kernel. The failure is evidence that EOS
closure does not replace upstream state validation. Contrast this once with the synthesis
fixed-density path, which floors that nuclei-density difference at `1e-300`; do not present the
two safeguards as an identical policy.

**Exact transition to 3.8.**

> One element makes the feedback visible, but a stellar mixture contains ninety-nine elements
> whose charges must be summed in a fixed order at every depth.

### 3.8 Ninety-nine elements close one depth; depths are independent

For one depth, give the exact causal order:

```text
trial electron density
→ nuclei density from P/(kT) - n_e
→ Z = 1…99 in ascending order
→ mode-12 stage fractions
→ abundance-scaled actual populations
→ retained-stage charge sum
→ damped update
```

The element scale is

\[
n_{s,r}
=
n_{\rm nuclei}\,A_s\,f_{s,r},
\]

where \(A_s\) is the linear abundance already established in Chapter 2. Do not rederive dex
notation.

Explain two exact qualifications before reporting conservation:

1. helper ion stages can participate in normalization without being returned, so a stored subset
   can have a nonzero truncation deficit;
2. the populations evaluated immediately before the final damped update can be one evaluation
   behind the final \(n_e\). The atmosphere's subsequent population schedule refreshes the packed
   state; the synthesis atom-only full solver does not perform that extra refresh.

**Visible cell C9 — complete one-depth atmosphere closure.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| one row of the atom-only closure fixture, atmosphere tables, 99-element abundance row, positive seed | final `electron_density`, `total_nuclei_number_density`, `charge_square_density`, `mass_density`, refreshed actual and normalized packed rows | canonical atmosphere CPU NumPy/Numba `float64` path |

The output must report:

- particle identity \(P/(kT)=n_{\rm nuclei}+n_e\);
- retained-stage charge residual under the declared tolerance;
- element-population sums and helper-stage truncation deficits for selected cool and hot elements;
- proof that the final full schedule refreshed the reusable population arrays.

Load no golden yet; C10 will compare the same multi-depth state after both scalar and parallel
paths have been computed.

Now ask whether one depth needs another. It does not for this atom-only closure: every row owns its
entire fixed-point orbit. Reuse Chapter 2's `prange` knowledge without reteaching decorators.

One short exact-source card, at most 12 lines, should show only:

```text
for layer_index in numba.prange(layer_count):
    solve the complete scalar layer
```

The prose must explain why this boundary is safe: no row writes another row and there is no
cross-depth reduction. It must also say that `saha_partition_depth_batch` is compiled but serial
in depth.

**Visible cell C10 — scalar-depth, one-thread, and multi-thread parity.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| 4–8-depth atom-only fixture spanning cool, solar-like, and hot conditions | repeated scalar-depth state, `prange` state at one and several declared thread counts, warm timings, exact differences | canonical atmosphere Numba CPU `float64`; subprocess or controlled thread harness records versions and thread counts |

Compare:

- multiple one-depth scalar calls;
- multi-depth `prange` at one thread;
- multi-depth `prange` at the declared multi-thread count;
- `chapter03_atmosphere_atomic_state.npz`, loaded only after all local results exist.

The output must name each compared field, the maximum absolute difference, whether exact equality
was measured, Numba version, thread count, and cold/warm state. Do not infer bit identity from
`prange`; report what the run measured.

**Required output interpretation.** If equality is exact, attribute it to each row retaining its
own arithmetic order, not to parallelism in general. If a platform differs, report the measured
field and tolerance without weakening the contract silently.

Use a concise contrast table:

| route | work over depth | element order | stopping policy |
| --- | --- | --- | --- |
| atmosphere atom-only | `prange`, one complete fixed point per depth | ascending \(Z\) inside each row | each depth exits independently |
| synthesis atom-only | device-batched depth algebra for one element at a time | ascending Python element loop | all rows continue until the worst row converges |
| molecule-enabled atmosphere | deferred to Chapter 4 | coupled depth continuation | not the atom-only `prange` route |

**Exact transition to 3.9.**

> Charge closure gives trustworthy particle counts. Opacity and line calculations will also need
> mass per volume, collision partners, and a velocity-width scale, all of which can now be derived
> without guessing new populations.

### 3.9 From particle counts to density and broadening support

First derive mass density:

\[
\rho=n_{\rm nuclei}\,\bar m_{\rm nuclei}\,m_u.
\]

State the exact path-specific convention:

- atmosphere `compute_mean_nuclear_mass_amu` takes the direct abundance-weighted mass and expects
  the deck representation to be on the intended number-fraction scale;
- synthesis normalizes by the abundance sum when it must infer the mean mass;
- supplying the atmosphere `mass_density` to a fixed bridge preserves the atmosphere result.

Do not merge these into one cleaned-up definition.

Then introduce the ordinary-line neutral-collision proxy:

\[
n_{\rm pert}
=
\left(n_{\rm H\,I}
+0.42n_{\rm He\,I}
+0.85n_{\rm H_2}\right)
\left(\frac{T}{10^4\ {\rm K}}\right)^{0.3}.
\]

Explain that it is a calibrated proxy, not total neutral density. In this chapter
\(n_{\rm H_2}=0\); Chapter 4 will earn the molecular term.

**Visible cell C11 — mass density and the atom-only perturber proxy.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| C10 actual H I and He I populations, nuclei density, atmosphere mean nuclear mass, temperature, zero H2 | `mass_density (D,)` g cm\(^{-3}\), `neutral_collision_density_proxy (D,)` cm\(^{-3}\) | canonical CPU NumPy `float64` functions |

Print direct arithmetic checks and selected cool/hot layer values. Predict that setting the neutral
populations to zero makes the proxy zero and that increasing temperature at fixed populations
raises only the explicit \(T^{0.3}\) factor.

**Required output interpretation.** Read the maximum arithmetic differences. State that the H2
coefficient is present in the exact formula but its Chapter 3 input is honestly zero.

Now derive the fractional Doppler support:

\[
\frac{\Delta v_{\rm D}}{c}
=
\frac{1}{c}
\sqrt{\frac{2k_{\rm B}T}{m}+\xi^2}.
\]

Define \(\xi\) as atmosphere `microturbulence` in cm s\(^{-1}\), not the public initializer label
`microturbulence_km_s`.

Predictions:

- at \(\xi=0\), lighter elements have larger thermal width;
- when \(\xi^2\) dominates, masses matter less;
- widths are positive and dimensionless.

**Visible cell C12 — Doppler limits and exact support arrays.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| 2–4 temperatures, two declared masses, two microturbulence limits, C10 packed state | hand-computed \(v_D/c\), atmosphere `(D,1006)` widths and population-over-density-width factors, synthesis `(D,6,139)` widths | canonical CPU NumPy `float64` paths |

Print a compact limit table rather than a fourth plot. Check the atmosphere final packed slot and
synthesis non-atomic columns remain zero. State the path-specific mass convention: major-isotope
mass per packed atmosphere slot versus element mass in synthesis column building.

**Required output interpretation.** Read the light/heavy ratio in the thermal limit and its change
when microturbulence dominates. Stop at the fractional support quantity; Chapter 6 owns frequency
widths and line-profile consequences.

**Exact transition to 3.10.**

> Counts and masses determine how matter moves and collides, but ionization also stores energy.
> The closed atomic state must account for that reservoir before it can support later
> thermodynamics.

### 3.10 Where ionization energy is stored

Build the atomic energy density from three pieces:

\[
u_{\rm atom}
=
\frac{3}{2}(n_e+n_{\rm nuclei})k_{\rm B}T
+
\sum_j n_j
\left[
E_{{\rm ion},j}
+k_{\rm B}T\frac{\partial\ln U_j}{\partial\ln T}
\right].
\]

Define:

- translational kinetic energy;
- cumulative ionization energy already paid by each stage;
- excitation response of the partition function.

Then divide by `mass_density` to obtain `specific_internal_energy` in
erg g\(^{-1}\).

At the exact boundary, explain:

- cumulative ionization energies come from the packed 999-entry potential table;
- partitions are evaluated at `T * 1.001` and `T * 0.999`;
- the exact symmetric approximation is
  `(U_plus - U_minus) / (U_plus + U_minus) * 1000`;
- packed slots 0–839 are summed in increasing order;
- the helper returns an array and does not automatically mutate
  `state.specific_internal_energy`;
- molecular internal energy is not included.

**Visible cell C13 — atomic specific internal energy.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| C10 refreshed packed populations, temperature, nuclei/electron/mass densities, exact partition and ionization tables | translational check, exact atomic `specific_internal_energy (D,)`, selected component diagnostics | canonical atmosphere CPU NumPy/Numba `float64` |

Predict positivity for the declared physical fixture and that the full value is at least the
translation-only contribution. Compare a reader-computed translation term with the exact total,
then compare the locally computed exact total to its golden only after evaluation.

**Required output interpretation.** Read the actual translation fraction in one cool and one hot
layer. State that the difference contains ionization and partition-excitation energy; do not call
it purely ionization energy. Mention once that Chapter 12 will use EOS energy responses in
convection, without teaching that later calculation.

**Movement-II bridge.**

> We now have a physically interpretable atomic state at every depth. The remaining problem is
> representational: the CPU atmosphere solver and device synthesis engine store that same state in
> different exact layouts, and a careless reshape can exchange species, ion stages, or physical
> meanings.

## Movement III — Move the state without changing its claim

### 3.11 Two exact layouts for two engines

Begin from use, not history:

- atmosphere iteration needs a sparse fixed-column packed state with width 1006;
- synthesis needs a regular ion-stage/species cube;
- neither is a universal physics layout.

Introduce the atmosphere state:

```text
ion_stage_populations_by_packed_slot                 (D,1006), actual cm^-3
partition_normalized_populations_by_packed_slot      (D,1006), cm^-3 per U
fractional_doppler_widths                            (D,1006), dimensionless
```

Explain that 1006 is a sparse interface, not 1006 consecutive atomic stages. The mode-12 and
mode-11 schedules differ deliberately. Give the compact schedule facts only after the decoder is
understood:

- 198 atom-only jobs;
- 356 populated actual slots;
- 403 populated partition-normalized slots;
- atomic range 0–837;
- intentional gaps;
- molecule-enabled later slots, including H2 at slot 840, remain deferred.

One short exact-source card, at most 15 lines, may combine
`atomic_population_slot_start` and the essential line of `decode_population_code`. Explain that
codes such as `1.01`, `2.02`, and `20.09` are fixed source constants, not an invitation to construct
arbitrary decimals.

**Visible cell C14 — decode and audit the packed schedule.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| exact atom-only population job schedule | decoded examples, slot starts, job count, populated-slot counts, overlap/gap diagnostics | canonical CPU Python/NumPy metadata path |

**Required output interpretation.** Read the actual counts and identify one intentional difference
between actual and partition-normalized output counts, such as Ca–Ni. Do not recite the entire
198-job table.

Now introduce all exact synthesis axes:

```text
EOSResult internal:       (D,99,6) = depth, element, stored ion stage
public schema cube:       (D,6,139) = depth, ion stage, species
```

State that six is storage capacity, not a promise that every element has six meaningful stages.
In the atom-only route, species columns 0–98 receive atomic values and
99–138 remain empty. This empty tail is not the molecular address rule:
Chapter 4 derives selected stage-5 destinations from line-list species codes
across the public species axis.

Place the second original schematic before mapping code.

**Schematic S2 — sparse packed state to public cube.**

- Landscape, original textbook composition.
- Left: a long sparse strip labeled `(D,1006)` with highlighted H, He, Fe, gaps, and an unmapped
  tail.
- Middle: an explicit slot-map funnel with only some arrows.
- Right: a six-row by 139-column slice labeled `(D,6,139)`, with the first 99
  species region and an unused atom-only tail spanning 99–138.
- Highlight one H, one He, and one Fe destination.
- Do not imply every packed slot maps or that a raw reshape performs the conversion.
- Caption must call it conceptual and state the exact axis meanings.

**Visible cell C15 — sentinel mapping before physical values.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| interface-only unique integer sentinels in three `(D,1006)` arrays | three `(D,6,139)` cubes and exact-location diagnostics | canonical `_packed_atomic_cube` mapping, CPU NumPy `float64` |

Check:

- H, He, C, Mg, Al, Si, and Fe destinations;
- zero unmapped cells;
- actual and normalized arrays are not swapped;
- species columns 99–138 remain zero.

**Required output interpretation.** Read at least three exact source-to-destination sentinels.
State that this proves interface routing, not physical correctness.

**Visible cell C16 — map the refreshed physical state.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| C10 physical packed actual, normalized, and Doppler arrays plus exact physical columns | public `ion_stage_populations`, `partition_normalized_populations`, `fractional_doppler_widths`, and named H/He/C/Mg/Al/Si/Fe fields | canonical `prepare_structured_handoff_population_state` with `molecules_enabled=False`, then the packed-to-cube bridge, CPU NumPy `float64` |

Compute first, then load `chapter03_packed_bridge_outputs.npz`. Verify source/fixture hashes before
array comparison.

The output must include:

- exact cube axes and dtypes;
- named-field equality to the correct packed slots;
- zero unused atom-only tail;
- separate actual and normalized population identities;
- unchanged supplied `electron_density` where the packed handoff is fixed.

**Required output interpretation.** State which fields matched exactly and which policy applies if
a tolerance is needed. Do not describe the bridge as a reshape.

**Exact transition to 3.12.**

> The mapping preserves values, but it does not decide whether the supplied electron density was
> physically closed. That decision belongs to the route used before the map.

### 3.12 Full closure or fixed-\(n_e\): choose the claim

Put the two routes side by side only after both are physically understood:

```text
full closure:
T, P, composition, seed
→ iterate populations and charge
→ solved electron_density + population state

fixed-n_e bridge:
T, P, composition, supplied electron_density
→ evaluate populations once at that density
→ exactly preserved electron_density + population state
```

Use exact names only here:

- `solve_electron_density`;
- `solve_population_state`;
- `solve_population_state_at_electron_density`;
- public `build_structured_atmosphere`.

Explain the exact public call chain and the potentially misleading
`electron_density_seed` name: in the structured column bridge, it is the supplied established
density passed to the fixed-\(n_e\) routine, not a hidden invitation to solve charge.

For this comparison set `molecules=False`. State why:

- the atom-only fixed bridge must preserve the supplied density exactly;
- molecule-enabled atmosphere ordering belongs to Chapter 4 and can update the runtime density
  state before a packed fill;
- this chapter may not generalize its atom-only preservation test across that deferred branch.

**Visible cell C17 — same columns, two claims.**

Contract:

| reads | writes | execution |
| --- | --- | --- |
| same controlled `temperature`, `gas_pressure`, abundance, mass policy, and supplied/seed density | full-closure `PopulationState`, fixed-\(n_e\) `PopulationState`, charge residuals, density changes, population checks | canonical synthesis CPU `torch.float64` plus host NumPy `float64` closure |

Compute both routes before loading
`chapter03_synthesis_atomic_state_cpu_float64.npz`. Present a compact claim table:

| question | full closure | fixed-\(n_e\) |
| --- | --- | --- |
| may `electron_density` move? | yes | no |
| tests charge conservation? | yes, under declared retained-stage tolerance | no |
| computes atomic populations? | yes | yes |
| preserves supplied density exactly? | not its purpose | required |

Report:

- the full route's maximum retained-stage charge residual;
- exact equality of fixed output and supplied `electron_density`;
- particle and population diagnostics;
- a one-element abundance perturbation showing that the full synthesis route applies the requested
  abundance once during charge accumulation and population assembly, rather than applying it
  inside `populations` and then a second time;
- the one-evaluation-behind nuance for synthesis full closure;
- golden comparison after local calculation.

**Required output interpretation.** Do not call the fixed route less accurate in general. Say that
it makes a different claim: it trusts an electron density already established by an upstream
physical atmosphere and must not change it.

Explain one exact wrapper limitation in prose: public `eos_tolerance` is forwarded through the
column-builder layers but does not control the atom-only fixed-density fill; it must not be cited
as evidence of a hidden charge solve.

**Visible cell C18 — declared backend tolerance profile.**

This is a compact reproducibility cell, not a speed chart.

Contract:

| reads | writes | execution |
| --- | --- | --- |
| the same C17 fixture and manifest-bound tables | CPU `float64` reference comparison; CUDA `float64` and MPS `float32` comparisons only when available; explicit unavailable status otherwise | exact runtime-selected backends |

For each backend report device, dtype, Torch version, compared fields, maximum absolute and
relative difference, and the backend-specific acceptance tolerance. Never substitute one universal
tolerance or claim that an unavailable backend passed.

**Required output interpretation.** Read the actual available-device result. Keep the scientific
claim limited to population-state parity under the stated policy; performance and full synthesis
belong later.

**Exact output contract at the end of 3.12.**

The reader now has:

| exact output | shape | unit / meaning |
| --- | --- | --- |
| atmosphere actual packed populations | `(D,1006)` | cm\(^{-3}\) |
| atmosphere partition-normalized packed populations | `(D,1006)` | cm\(^{-3}\) per partition function |
| synthesis internal EOS arrays | `(D,99,6)` | partitions or stage fractions as named |
| public `ion_stage_populations` | `(D,6,139)` | actual cm\(^{-3}\) |
| public `partition_normalized_populations` | `(D,6,139)` | cm\(^{-3}\) per partition function |
| `electron_density`, `total_nuclei_number_density` | `(D,)` | cm\(^{-3}\) |
| `mass_density` | `(D,)` | g cm\(^{-3}\) |
| `fractional_doppler_widths` | packed or public layout | dimensionless \(v/c\) |
| atomic `specific_internal_energy` | `(D,)` | erg g\(^{-1}\) |

The prose must attach one of two claim labels to any public population state:

- **charge-closed atom-only state**, with a declared residual and stopping policy; or
- **fixed-\(n_e\) population state**, with exact preservation of its supplied density.

**Exact transition to 3.13.**

> We can now answer how an atom-only gas divides nuclei among levels and ions, and we can state
> whether its electron density was solved or preserved. One physical assumption remains exposed:
> every nucleus is still assigned to an atom or ion.

### 3.13 Chapter summary

Return to the opening cool and hot layers. Answer only what has been computed:

- Boltzmann weights distribute one ion stage among levels.
- Partition functions normalize those levels and define the reusable \(n/U\) quantity.
- Saha ratios distribute an element among ion stages for a trial \(n_e\).
- particle and charge conservation create a damped electron-density fixed point;
- the closed populations determine nuclei/mass density, collision and Doppler support, and atomic
  specific internal energy;
- exact maps carry actual and partition-normalized populations between sparse atmosphere and
  public synthesis layouts;
- full closure and fixed-\(n_e\) fills make different, explicitly tested claims.

The summary contains no source-trap details, new field names, or new equations.

State the unresolved limitation:

> Atomic conservation assumes every nucleus remains in an atom or ion. In cool layers, molecules
> bind several elements at once, so the separate element budgets become a coupled equilibrium
> problem.

Use the required causal navigation:

### Next: let atoms bind into molecules

> [Chapter 4: Molecules and Coupled Equilibrium](/reader.html?ch=4) replaces the atom-only
> assumption with mass action and simultaneous elemental conservation. It must preserve the
> atomic distinctions built here while solving for species that consume several elements at once.

## Cell, figure, and pacing contract

### Visible computational density

The accepted post-audit target is exactly 21 substantial visible executable
cells. The increase is purposeful: timing versus parity, Doppler physics versus
packed support, full/fixed route semantics versus abundance perturbation, and
the two stages of the energy derivation each require separate evidence.

| movement | cells | purpose |
| --- | ---: | --- |
| I — one atom | C1–C6 | excitation, transparent Saha, then just-in-time production partitions and lowering |
| II — charge closure | C7–C15 | fixed point, 99-element closure, separate timing/parity, density, Doppler/support, derived energy |
| III — interfaces and claims | C16–C21 | schedule, map, full/fixed route, abundance-once probe, backend policy |

In addition, at most two focused exact-source cards may be visible:

1. ordered `ground_partition_value`, no more than 15 lines;
2. the `prange` outer boundary or the packed decoder, no more than 15 lines.

If both the `prange` boundary and packed decoder need source cards, omit the ground-function card
and teach it through the canonical call and output. The audited chapter must
not exceed 21 substantial code cells or three short source cards without
triggering a new density review.

Mechanical imports, style setup, and path resolution may live in one generated hidden setup cell.
No scientific computation, manifest result, prediction, or parity result may be hidden there.

Every visible executable cell:

- has one conceptual purpose;
- targets 10–30 lines;
- is preceded by reads/writes/shape/unit/dtype/device prose;
- has a visible result;
- is followed immediately by an interpretation using actual values;
- is separated from the next code cell by a prose bridge.

### Quantitative figures

Use exactly three required one-panel figures:

1. C1: two-level excited/ground ratio versus temperature;
2. C5: H I and H II fraction versus temperature at fixed \(n_e\);
3. C8: electron fixed-point residual versus iteration.

The Doppler limit remains a compact numerical table unless a rendered audit shows that a plot
materially improves understanding. Adding it would require removing another quantitative figure or
reopening the density review.

All plots use the shared professional style:

- white background;
- paper-inspired serif mathematical typography;
- inward ticks;
- dark reference curves and color-safe blue/orange accents;
- axis labels with quantities and units;
- direct labels when they reduce legend burden;
- no multi-panel dashboard, unreviewed default colors, or decorative grid.

The paragraph after each plot must read actual values from the generated result.

### Original schematics

Use exactly two original textbook-owned schematics:

1. S1: level ladder → partition sum → ion ladder ↔ charge closure;
2. S2: sparse `(D,1006)` atmosphere layout → explicit map → `(D,6,139)` public cube.

Each needs:

- an owned entry in `scripts/textbook_schematic_specs.py`;
- generation provenance and SHA-256 in the schematic manifest;
- scientifically reviewed labels and arrows;
- alt text and a caption that calls the image conceptual;
- inspection at notebook width and in the local reader.

The official website supplies aesthetic guidance only. No website asset or composition is reused.

### Timing estimate and pause points

The intended 90-minute lecture/lab rhythm is:

| movement | minutes | natural pause |
| --- | ---: | --- |
| I | 30 | after a supplied \(n_e\) produces density-corrected ion stages |
| II | 35 | after a multi-depth, charge-closed atomic state and support bundle |
| III | 20 | after exact mapping and full/fixed claim comparison |
| summary | 5 | causal handoff to molecules |

The reader page should expose anchors at all three movement headings. If the executed narrative
cannot retain these timings without skipping a derivation or output interpretation, first shorten
policy inventory prose and move exhaustive tables to reference material. Do not delete physical
derivations, checks, or route distinctions. A chapter-count change requires the global density gate,
not an ad hoc split.

## Data and parity placement

Goldens never produce the reader's result. They appear only after the corresponding local
calculation:

| local computation | comparison loaded afterward |
| --- | --- |
| C6 scalar/batch Saha modes | `chapter03_atmosphere_saha_outputs.npz` |
| C9–C10 complete atmosphere state | `chapter03_atmosphere_atomic_state.npz` |
| C13 atomic energy | the energy field in the atmosphere-state golden |
| C15–C16 exact bridge | `chapter03_packed_bridge_outputs.npz` |
| C17 synthesis full/fixed state | `chapter03_synthesis_atomic_state_cpu_float64.npz` |
| C18 optional device states | backend-specific tolerance profiles, never the CPU tolerance reused blindly |

The five fixture roles remain explicit:

- synthetic two-level atom;
- exact one-element source-branch fixture;
- atom-only closure fixture;
- interface-only sentinel fixture;
- support-quantity fixture or manifest-bound slice of the closure result.

The synthesis source bundle must be split so static EOS tables and the bundled 80-depth computed
state do not share one textbook data role. Every local product records source commit, source-file
hash, per-array hashes, shapes, dtypes, units, and builder command.

## Production-boundary boxes

Use at most four concise boundary boxes, each after the transparent core:

1. **Partition policies are stage-specific.** Special, packed, PFIRON, ground, and occupation
   policies; no universal table stack.
2. **The two Saha ladders are not identical algorithms.** Direct-ratio atmosphere versus
   log-space synthesis; helper-stage and stopping differences.
3. **Parallelism follows physical independence.** Atom-only depth `prange`; ordered elements;
   molecule-enabled continuation deferred.
4. **Full closure and fixed-\(n_e\) make different claims.** One solves charge; one preserves an
   upstream density.

Do not scatter implementation-trap callouts through every paragraph. The source contract remains
the exhaustive audit reference; the chapter teaches only the differences needed to interpret its
own outputs.

## Paragraph and neighbor audit questions

Before acceptance, read Chapter 2's final population-bridge paragraphs, all of Chapter 3, and
Chapter 4's opening together. Confirm:

- Chapter 3 begins from Chapter 2's empty-but-named population fields without repeating schema
  inventory or abundance pedagogy.
- `electron_density` is not called converged until charge closure has been measured.
- a fixed-\(n_e\) route never receives credit for charge closure.
- level population, actual ion-stage population, ion fraction, and population divided by
  partition function remain distinct.
- the source's one-based helper stages and the mathematical neutral-first ladder are never mixed
  silently.
- stored stage sums are reported with truncation deficits, not forced to one.
- the atmosphere and synthesis table stacks remain separate and no cross-stack equality is
  promised.
- the atom-only atmosphere `prange` boundary is shown because depths are physically independent,
  not because parallel code is fashionable.
- all molecule-specific residuals, Jacobians, continuation, and internal energy remain in Chapter
  4.
- the H2 term in the collision proxy is present but zero, with one forward promise and no
  molecular derivation.
- Chapter 4 opens by consuming the atomic state, not re-teaching Boltzmann, partition functions,
  Saha, or the actual-versus-normalized distinction.

For every paragraph, ask which unresolved question it answers or creates, which equation/cell it
prepares, which result it interprets, or which limitation it makes honest. Remove any paragraph
that does none of these.

## Chapter-level acceptance gate

The causal draft is ready for canonical implementation only if all answers are yes:

- Does the opening make an empty schema field physically unsatisfying before introducing EOS
  machinery?
- Does one atom lead naturally from excitation to partition normalization to ionization?
- Are the real partition recipes introduced only after the two-level sum is understood?
- Are the atmosphere and synthesis stacks taught as separately exact rather than forcibly unified?
- Does the Saha plot state its fixed-\(n_e\) condition before claiming a temperature trend?
- Is the electron solver called a damped fixed point, never Newton?
- Does the 99-element state report helper-stage truncation and final-population staleness honestly?
- Does the `prange` section apply Chapter 2 knowledge without repeating the tutorial?
- Are mass, collision, Doppler, and energy support introduced only after actual populations exist?
- Are `(D,1006)`, `(D,99,6)`, and `(D,6,139)` always accompanied by their axis meanings?
- Does a sentinel map precede physical bridge parity?
- Does the fixed-\(n_e\) path preserve input exactly with molecules disabled?
- Are goldens loaded only after local computation and bound to the exact fixture/table identities?
- Are unavailable backends reported as unavailable rather than silently passed?
- Are there no detached exercises, source dumps, invented public wrappers, or legacy framing?
- Does the summary add no new concept and make molecules the one necessary next dependency?
