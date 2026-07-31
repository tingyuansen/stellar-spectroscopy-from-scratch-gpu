# Chapter 7 first-pass contract — Atomic Line Forests and Special Profiles

Status: bounded reader-facing design; no implementation or publication authority  
Pinned Payne Zero commit: `9c44001feae40b85146630499e6f8a5fed42e5af`  
Audience: final-year undergraduate / first-year graduate student  
Provisional title: **From One Line to an Atomic Forest**

## 0. Canonical placement and non-negotiable boundaries

This contract follows the canonical fifteen-chapter architecture in `PLAN.md`
and `design/global_chapter_contracts.md`.

```text
Chapter 6: one trustworthy ordinary atomic line
    -> Chapter 7: full atomic catalogs, sparse accumulation, special profiles
    -> Chapter 8: molecular bands and their source-specific compilation
    -> Chapter 9: radiative transfer with scattering
    -> Chapter 10: GPU synthesis as a complete performance system
```

There is no architectural conflict to reconcile. Chapter 7 is not a transfer
chapter and must not compute an emergent spectrum. It produces checked atomic
line-opacity slabs.

Chapter 6 already owns and has checked:

- the ordinary LTE line-strength equation;
- the Boltzmann/FASTEX evaluation used by that equation;
- fractional Doppler width;
- the three-term radiative, Stark, and van der Waals damping sum;
- the Harris/Voigt center and wing profile;
- one-line cutoff and deposition.

Chapter 7 may invoke those pieces by exact name, but it must not rederive them.
Its new work is to preserve source semantics while scaling from one line to a
heterogeneous forest and to explain why a few line families require distinct
profile routes.

Chapter 8 owns diatomic, TiO, H2O, and H3+ source-specific decoding,
corrections, selection, and compilation. Chapter 7 may identify the common
atmosphere `SelectedLineCatalog` boundary that those families eventually use,
but it must not preview their individual rules.

The chapter has two visible movements:

1. **7A — The ordinary atomic forest:** decode, correct, window, select, map,
   and deposit many ordinary records without a dense line axis.
2. **7B — Special atomic profiles:** route H, D, He, and autoionizing records;
   state exactly what happens to COR and PRD tags; merge high hydrogen series
   into pseudo-continuum opacity.

Keep these movements in one chapter by default. Split only if the implemented
notebook exceeds 18 substantial visible code cells, cannot fit a 90-minute
lecture/lab, or cannot preserve separate ordinary-forest and special-profile
checkpoints without rushing.

There are no detached exercises. Every useful calculation, diagnostic, and
limit check belongs in the main causal narrative.

## 1. The chapter's single question

Open with the ordinary Fe I line completed in Chapter 6. Then place it beside a
short wavelength interval containing many overlapping lines. Do not begin with
a file format or a directory tree.

Ask:

> A single line can be evaluated carefully. How can millions of heterogeneous
> atomic records be reduced to the lines that matter, routed to the correct
> physics, and added to one opacity grid without losing weak wings or
> double-counting overlapping pixels?

The chapter's destination is the following self-contained construction:

```text
source records
    -> corrected, source-faithful atomic catalog
    -> wavelength and atmosphere-dependent keep decisions
    -> ordinary and special route masks
    -> sparse center and wing deposits
    -> one checked atomic line-opacity slab
```

The opening should make the scaling problem numerical. A hypothetical dense
profile tensor has shape `(D, L, W)`. Even a modest example with 80 depths,
one million lines, and 50,000 wavelength samples would require
\(4\times10^{12}\) float32 values, or about 16 TB. Almost all entries would be
zero or negligible. The reader should therefore expect a sparse algorithm
before seeing its implementation.

Do not repeatedly invoke the name Payne Zero in the prose. Introduce exact
production names only when the reader has built the corresponding physical or
numerical object.

## 2. Reader promise and prerequisites

By the end of the chapter, the reader should be able to explain and reproduce:

- why a source wavelength, an energy-derived wavelength, and a grid index are
  different objects;
- how isotope and energy corrections alter a catalog record before selection;
- why line-size margins are part of the physics-preserving window contract;
- the exact atmosphere catalog-selection inequality, including every strict
  and non-strict comparison;
- why the atmosphere selector uses `numba.njit` and `numba.prange`, and what
  changes between ordinary Python, compiled serial code, and compiled parallel
  code;
- why synthesis performs a second, local keep decision at each depth and line;
- how center and wing contributions reach a `(D,W)` slab without allocating
  `(D,L,W)`;
- why scatter-add, per-thread private buffers, and fixed reduction order solve
  related but not identical problems;
- which route code owns ordinary, PRD-tagged, autoionizing, COR, H/D, and He
  records;
- why hydrogen fine structure, Stark broadening, neighboring-series
  competition, and series merging cannot be represented by one ordinary
  Voigt line;
- which products are invariant across atmospheres and which must be recomputed
  for each atmosphere.

Assume only basic mechanics, thermodynamics, calculus, arrays, and ordinary
Python. Define “catalog,” “record,” “mask,” “race,” “chunk,” and “reduction”
when first used. A reader should not need prior knowledge of Numba, Torch, GPU
programming, fixed-width spectroscopy files, or Kurucz implementation history.

## 3. Exact inputs and outputs

Use \(D\) for depth, \(W\) for wavelength samples, and \(L\) for catalog
records only while explaining shapes. The executable names remain the exact
source names.

### 3.1 Supplied physical state

Chapter 7 consumes the same physical arrays that Chapter 6 used for one line.
One compact table is enough; do not repeat their derivations.

| supplied object | exact name | shape | unit | role here |
| --- | --- | --- | --- | --- |
| partition-normalized ion populations | `partition_normalized_populations` | synthesis `(D,6,139)` | cm\(^{-3}\) | gather each line's species and ion stage |
| population divided by density and Doppler width | `partition_normalized_population_over_mass_density_and_fractional_doppler_width` | atmosphere `(D,S)` | contract-declared composite | conservative source selection and standard-line opacity |
| fractional Doppler widths | `fractional_doppler_widths` | synthesis `(D,6,139)`; atmosphere `(D,S)` | dimensionless | reuse Chapter 6 width |
| mass density | `mass_density` | `(D,)` | g cm\(^{-3}\) | number-to-mass conversion |
| electron density | `electron_density` | `(D,)` | cm\(^{-3}\) | Stark widths and H series merging |
| neutral collision proxy | `collision_density_proxy` | `(D,)` | cm\(^{-3}\) | ordinary and He collisional damping |
| excitation conversion | `hc_over_kt` | `(D,)` | cm | FASTEX/Boltzmann lookup input |
| smooth comparison opacity | `continuum_opacity` or `continuum_line_selection_threshold` | `(D,W)` or `(D,B)` | cm\(^2\) g\(^{-1}\) | decide whether a line can matter |
| geometric wavelength samples | `wavelength_grid` / `opacity_wavelength_grid_nm` | `(W,)` | nm | center and wing targets |

The molecular-hydrogen population enters the hydrogen broadening state as a
perturber. That fact does not move molecular line opacity into this chapter.

### 3.2 Catalog products introduced here

The synthesis lane earns a structure-of-arrays `LineCatalog`. A
**structure of arrays** stores one contiguous array per field rather than one
Python object per line. This lets the implementation select or transfer one
field for many lines without constructing millions of small objects.

The visible interface table must include every `LineCatalog` field:

| field group | exact fields | meaning |
| --- | --- | --- |
| wavelength | `wavelength_nm`, `index_wavelength_nm` | corrected physical wavelength and the wavelength used for grid indexing |
| strength/excitation | `oscillator_strength`, `log_oscillator_strength`, `lower_excitation_cm`, `classical_line_strength` | source strength, lower energy, and Chapter 6 strength prefactor |
| normalized damping | `radiative_damping`, `stark_damping`, `van_der_waals_damping` | \(\gamma/(4\pi\nu)\)-convention coefficients for ordinary profiles |
| raw damping provenance | `raw_radiative_damping_log`, `raw_stark_damping_log`, `raw_van_der_waals_damping_log` | unsmoothed source meanings needed by special routes |
| species | `atomic_number`, `ion_stage`, `species_code` | population lookup identity |
| routing | `line_size`, `line_type` | window-margin and profile-route decisions |
| hydrogen levels | `lower_principal_quantum_number`, `upper_principal_quantum_number` | resolved-line and merged-series identities |
| bookkeeping | `type_segments`, `grid`, `sort` | ordering and grid provenance |

The atmosphere's compact family-independent product is
`SelectedLineCatalog`, with the exact fields
`packed_wavelength_index`, `packed_species_slot`,
`lower_excitation_index`, `log_strength_index`,
`radiative_damping_index`, `stark_damping_index`, and
`van_der_waals_damping_index`. All are one-dimensional arrays of equal length;
`line_count` reports that length. This representation contains selected
standard records, not every special-profile field.

The atmosphere's detailed special-record interface is
`LineTransitionCatalog`. Its visible fields are
`vacuum_wavelength_nm`, `lower_excitation_cm`, `oscillator_strength`,
`lower_hydrogen_level`, `upper_hydrogen_level`, `packed_species_slot`,
`line_type`, `hydrogen_continuum_selector_index`,
`continuum_species_slot`, the three damping arrays,
`packed_wavelength_index`, and `line_limit`.

Do not pretend these two atmosphere catalogs are interchangeable. The compact
catalog is designed for the enormous common selected-line stream; the detailed
catalog carries the information needed by special branches.

### 3.3 Chapter output

Both lanes produce an additive mass-absorption slab:

| exact product | shape | unit | dtype/meaning |
| --- | --- | --- | --- |
| `line_mass_absorption_coefficient` | `(D,W)` | cm\(^2\) g\(^{-1}\) | atmosphere and synthesis ordinary/He deposits use float32; synthesis H uses high precision on CPU/CUDA and float32 on MPS, so adding H may promote the composite slab |
| `selected_line_count` | scalar | count | atmosphere count of records whose center strength passed at least one sampled depth gate; it is not a count of nonzero pixels |
| route counts | small mapping | count | audit of decoded types before opacity |

The synthesis pipeline applies stimulated emission inside each atomic/H/He
branch before adding it to the shared line slab, which is initialized as
float32. The standard
atmosphere selected-line slab remains in its lane's established
pre-stimulation contract; downstream atmosphere transfer applies the reverse
process at its own boundary. The chapter must not compare these slabs as
though they were the same intermediate.

### 3.4 Exact implementation chain earned by the construction

Keep one compact interface map near the first use of each object:

```text
synthesis:
Grid -> parse_catalog/load_catalog -> precompute_invariants
     -> accumulate_atomic + accumulate_hydrogen

atmosphere:
source keep -> decode_selected_line_words -> SelectedLineCatalog
            -> accumulate_selected_line_opacity
detailed source -> LineTransitionCatalog
                -> accumulate_transition_line_opacity
```

The map is not a source-file tour. Each name appears only after its physical
or numerical object has been built in readable form. The teaching helpers
remain small enough that a student could reconstruct the chain without copying
the production functions.

## 4. A notation and decision ledger

Introduce symbols only where they shorten an explanation. Pair each symbol
with the exact code object.

| symbol | definition | exact implementation object |
| --- | --- | --- |
| \(R_{\rm grid}\) | constant resolving power of the internal geometric grid | `Grid.resolution` |
| \(r\) | \(1+1/R_{\rm grid}\) | `Grid.ratio` |
| \(j_l\) | discrete center or wing-anchor column of line \(l\) | host-computed center/anchor indices |
| \(P_{d,l}\) | partition-normalized population divided by density and fractional width | gathered population ratio |
| \(A^{(0)}_{d,l}\) | Chapter 6 strength before excitation | `pre_excitation_strength` |
| \(b_{d,l}\) | excitation weight | `excitation_weight` |
| \(A_{d,l}\) | \(A^{(0)}_{d,l}b_{d,l}\) | `line_amplitude` |
| \(C_{d,l}\) | local continuum-relative center cutoff | `center_cutoff` |
| \(K_{d,w}\) | final atomic line opacity | `line_mass_absorption_coefficient` |

Three keep decisions must never be collapsed into the word “selection”:

1. **window keep:** can a source record outside the requested interval have a
   wing inside it?
2. **atmosphere source keep:** could a packed source record matter anywhere in
   the current atmosphere?
3. **live depth-line keep:** does this already loaded synthesis record exceed
   the local opacity cutoff at this depth?

They occur at different stages, use different representations, and preserve
different things.

## 5. Movement 7A — Build the ordinary atomic forest

### 5.1 One trustworthy row becomes a catalog record

Begin with the Fe I row already used in Chapter 6, then place two deliberately
contrasting rows beside it: one with isotope/energy corrections and one with a
nonordinary route tag. Use a compact staged fixture, not a large source-file
listing and not a Markdown code block copied from production.

For each raw record, explain the causal correction order:

1. Apply the primary and secondary isotope corrections to the source
   log-strength when `apply_iso_corr=True`:

   \[
   \log(gf)=\log(gf)_{\rm raw}
   +\Delta_{\rm iso,1}+\Delta_{\rm iso,2}.
   \]

2. Take absolute term energies and identify the lower and upper values:

   \[
   \tilde E_l=\min(|\tilde E_1|,|\tilde E_2|),\qquad
   \tilde E_u=\max(|\tilde E_1|,|\tilde E_2|).
   \]

3. Decode the two energy-shift subfields in cm\(^{-1}\), then form

   \[
   \Delta\tilde E'
   =\left|(|\tilde E_2|+\delta\tilde E_2)
          -( |\tilde E_1|+\delta\tilde E_1)\right|.
   \]

4. If \(\Delta\tilde E'>0\), use

   \[
   \lambda_{\rm nm}=\frac{10^7}{\Delta\tilde E'}
   +10^{-4}\,\delta_{\rm isotope};
   \]

   otherwise retain `stored_wavelength_nm`.

5. Convert nonzero damping logs to linear coefficients. Fill missing values
   using the exact radiative, Stark, and van der Waals defaults. Normalize the
   ordinary-profile coefficients by `12.5664 * frequency`, the source's
   numerical \(4\pi\nu\) convention.

6. Preserve the raw damping logs alongside the normalized values. A
   source-specific field can have a different meaning in a special route;
   early normalization must not erase that provenance.

The reader should see a three-column before/after table for the tiny fixture:
raw field, corrected value, and why the change was necessary. The prose should
not tour byte offsets. Fixed-width parsing is an implementation detail after
the record semantics are understood.

Immediate checks:

- `oscillator_strength == 10**log_oscillator_strength`;
- `lower_excitation_cm <= upper excitation` in the explanatory table;
- a zero energy difference falls back to the stored wavelength;
- missing damping takes a finite default;
- raw damping logs survive unchanged.

### 5.2 A geometric grid needs context, not a shifted regeneration

Define the internal wavelength grid:

\[
\lambda_j=\lambda_0r^j,\qquad
r=1+\frac{1}{R_{\rm grid}}.
\]

Constant \(R_{\rm grid}\) means approximately constant velocity spacing. The
exact `Grid.build()` constructs float64 values. Discrete line-center and
wing-anchor decisions are also made from host float64 values before arrays
move to a device. This avoids device- or work-dtype-dependent flips in which
pixel receives a line.

The public requested grid is built exactly once. The synthesis lane grows
`WINDOW_CONTEXT_SAMPLES = 16` samples outward from each exact endpoint by
repeated division or multiplication by `Grid.ratio`. The internal context
grid's metadata bounds use `np.nextafter`; its `build()` result is not used to
regenerate the interior. Therefore:

```text
synthesis_wavelength_nm[output_slice] == requested_wavelength_nm
```

must be bitwise true.

Explain the need with one edge line: a line center just outside the requested
window can place a wing inside it. Context is discarded only after every
opacity source has been deposited.

### 5.3 Line-size codes become conservative wavelength margins

The line-size field determines how far outside a requested interval a record
may still be retained. State the exact rule before showing its vectorized
implementation.

Let

\[
s_l=\max(\texttt{line\_size}_l,0),
\quad
c_l=\operatorname{clip}\!\left(\min(8-s_l,7),1,7\right).
\]

All hydrogen-element records and the explicit H/D routes force \(c_l=1\). The
margin table, indexed by \(c_l-1\), is

\[
[100,\ 30,\ 10,\ 3,\ 1,\ 0.3,\ 0.1]\ {\rm nm}.
\]

The source's exact scale is

\[
q=
\begin{cases}
1,&\lambda_{\rm start}\leq500\ {\rm nm},\\
\lambda_{\rm start}/500,&\lambda_{\rm start}>500\ {\rm nm}.
\end{cases}
\]

Although the variable is named `red_window_margin_scale`, the selected margin
\(qm_l\) is used on both bounds. The inclusive keep test is

\[
\lambda_l\geq\lambda_{\rm start}-qm_l
\quad\text{and}\quad
\lambda_l\leq\lambda_{\rm end}+qm_l.
\]

Do not replace `>=` or `<=` with a prose approximation. Show two boundary
records that sit exactly on the blue and red limits and remain selected.

### 5.4 A route code is a physics decision

After corrections and windowing, classify records without changing catalog
order. Present this table before any special-profile equation:

| synthesis `line_type` | record family | current route |
| ---: | --- | --- |
| `0` | ordinary atomic line | Chapter 6 ordinary LTE/Harris route |
| `1` | `AUT` autoionizing | asymmetric Shore–Fano-like route |
| `2` | `COR` | parsed and retained for provenance; no standard opacity route |
| `3` | `PRD`-tagged | ordinary LTE/Harris route in the current implementation; no partial-redistribution transfer |
| `-1` | H I | hydrogen fine-structure/Stark/series route |
| `-2` | D I | current hydrogen route after isotope-aware source wavelength classification |
| `-3` | He I | helium special route |
| `-4` | He3 I | helium route with the exact isotope scaling |
| `-6` | He II | helium special route |

The distinction between a source tag and an implemented physical method is
important. In particular, `PRD` in the catalog does not mean that partial
frequency redistribution is solved. It is a retained label routed through
ordinary LTE opacity. `COR` must not be silently treated as ordinary opacity:
the synthesis invariant masks omit type 2, and the atmosphere detailed kernel
explicitly skips it.

The exact synthesis coupling must also be visible:
`accumulate_atomic(do_metal=True, ...)` invokes both `_accumulate_metal` and
`_accumulate_autoionizing`. “Metal” is therefore an implementation branch
name, not a chemically literal promise.

Ordering is part of the numerical contract. The standard synthesis pipeline
loads `sort="catalog"`, preserving source order. The optional
`sort="type_center"` performs a stable ordering by route and center column and
records contiguous `type_segments`. It may improve locality, but it also
changes float32 association; the chosen `sort` must therefore remain in cache
metadata and parity reports.

For the atmosphere detailed route, do not reuse synthesis codes where they do
not match. `LineTransitionCatalog` uses `-1` for resolved hydrogen. Remaining
nonnormal negative records encode merged-continuum information through their
numeric type value. Teach that lane at its actual interface rather than
inventing one universal encoding.

### 5.5 Invariants are separated from atmosphere-dependent decisions

Now earn `precompute_invariants`. Explain an **invariant** as a value that can
be computed once for a wavelength window and reused for many stellar
atmospheres:

- corrected wavelengths;
- population-column indices;
- host-float64 center and wing-anchor columns;
- classical line strengths;
- route masks;
- profile and exponential lookup tables.

The temperature, populations, density, collision partners, and continuum
opacity remain per-atmosphere state. Hydrogen's
`merge_wavenumber_by_depth` is the one atmosphere-dependent field in the
otherwise reusable synthesis `HydrogenInvariants` template and must be
replaced for every star.

The cache lesson should stay physical: cached products may accelerate reuse,
but cannot change selected values. The synthesis `LineCatalog` persistent
identity contains `CACHE_SCHEMA = 1`, `CACHE_LOGIC_VERSION = 3`, exact window,
resolution, sort,
source resolved path, source size, source modification time, and the isotope
correction switch. Corrupt or mismatched caches rebuild from the source.

### 5.6 The atmosphere makes one conservative source keep decision

The atmosphere may begin with roughly \(10^8\) packed source lines. It cannot
build a full line-by-depth matrix merely to decide which records are relevant.

First define the per-species, per-frequency-bin envelope:

\[
R_{s,b}
=\max_d
\frac{
  \texttt{partition\_normalized\_population\_over\_mass\_density\_and\_fractional\_doppler\_width}_{d,s}
}{
  \texttt{continuum\_line\_selection\_threshold}_{d,b}
},
\]

with zero returned where the denominator is not positive. This maximum is the
reason the test is conservative across depth: a record is not judged from one
arbitrarily chosen layer.

For each packed line, the implementation clips or overrides its species slot,
assigns its monotonized frequency bin, and clips its strength and excitation
lookup indices. For a standard atomic record, the zero-based population slot
is the clipped result of the exact integer operation
`abs(packed_species) // 10 - 1`. The frequency bin is
`max(bin_base, frequency_bin_floor)`, while both lookup indices are their
one-based packed values minus one and clipped to the lookup table. Then, in
float32 and in the exact written order, the kernel forms

\[
\texttt{center\_ratio}
=
\frac{0.026538}{1.77245}
\frac{
  \texttt{lookup\_float32[strength]}\ R_{s,b}
}{
  \texttt{frequencies[b]}
},
\]

using \(10^{-37}\) only when the frequency denominator is not positive, and

\[
\texttt{boltzmann}
=
\exp\!\left[
 -\texttt{lookup\_float32[excitation]}
  \ \texttt{deepest\_hc\_over\_kt}
\right].
\]

The record is kept if and only if all three clauses are true:

\[
\boxed{
R_{s,b}>\texttt{minimum\_population\_ratio}
\quad\land\quad
\texttt{center\_ratio}\geq1
\quad\land\quad
\texttt{center\_ratio}\,
\texttt{boltzmann}\geq1
}
\]

The first comparison is strict; the other two are inclusive. Include three
hand-built boundary records that isolate those comparisons. The result indices
are obtained with `np.where(mask)` and gathered into a
`SelectedLineCatalog`.

Native words generated inside the atmosphere route are decoded with
`detect_swapped_layout=False`. The public decoder retains
`detect_swapped_layout=True` only as an external compatibility path. The
chapter should show this as a representation boundary, not invite heuristic
layout detection on data whose provenance is already known.

This is the right point to teach Numba:

- ordinary Python interprets the loop body for every record;
- the exact production mask is `_selection_mask_compiled`;
- `@numba.njit(cache=True, nogil=True)` compiles a type-specialized machine-code
  loop and permits execution without the Python global interpreter lock;
- `parallel=True` plus `numba.prange` distributes independent line iterations
  across CPU threads;
- `prange` is safe here because each iteration writes only `mask[i]`;
- `fastmath` stays off because reassociation could change a borderline keep
  decision;
- the exponential is pretabulated with NumPy and gathered inside the compiled
  loop because a one-ulp difference in `exp` can flip a marginal record;
- the first call includes compilation time, while warm calls measure the
  reusable kernel.

One small timing cell should compare pure Python, warm serial `njit`, and warm
parallel `prange` on the same compact synthetic catalog. It must first assert
byte-identical masks. Report cold compilation separately and never imply that
parallel code is automatically faster for tiny arrays.

Clarify the three different kinds of reuse. `cache=True` may reuse compiled
Numba machine code. Full packed source arrays may remain resident in one
process under a resolved-path/size/mtime identity, and derived columns remain
tied to the lifetime of that array object. The selection output itself has no
disk cache: a new atmosphere solve recomputes the keep mask. Within one
atmosphere iteration sequence, the returned `SelectedLineCatalog` is reused
after the first iteration instead of regenerating the same selected words.

The chapter owns only the family-independent standard-line selector. It should
say once that Chapter 8 will reuse the boundary while supplying molecular
family corrections and overrides.

### 5.7 Synthesis makes a second, local depth-line decision

A line that survived a catalog window can still be negligible in most depths.
For ordinary type-0 and type-3 lines, reuse Chapter 6's strength definitions:

\[
A^{(0)}_{d,l}
=\texttt{classical\_strength}_l\,
  P_{d,l},
\qquad
A_{d,l}=A^{(0)}_{d,l}b_{d,l}.
\]

The exact local reference is

\[
C_{d,l}
=10^{-3}\,
\texttt{continuum\_opacity}
  [d,\texttt{metal\_center\_clamped}[l]].
\]

A depth-line pair reaches the center/profile work only when the population,
fractional width, and mass density are positive and

\[
A^{(0)}_{d,l}\geq C_{d,l},
\qquad
A_{d,l}\geq C_{d,l},
\qquad
A_{d,l}>0.
\]

Both strength tests are intentional. The pre-excitation comparison cheaply
rejects impossible records without making the later excited amplitude the
only guard. The second test is the physically relevant populated-line check.

Contrast the stages in one sentence: atmosphere source selection prevents an
enormous source stream from entering the opacity kernel; synthesis live
selection prevents an already loaded line from doing unnecessary work at a
particular depth.

The atmosphere's common selected-line kernel has one more earned shortcut.
Its production boundary requires exactly 80 layers and a population table
that includes packed slot 841. For each line it first checks every eighth
depth. An interior seven-depth block is visited only when one of its bounding
sample gates is active. `selected_line_count` increments when at least one of
those sampled gates passes, before the later Doppler-width check. A compact
cell should compare this eight-layer gate with the scalar source behavior on
the staged 80-layer fixture. Do not describe the returned count as the number
of deposited pixels or as a new physical observable.

### 5.8 Centers are a segmented sum, not a dense profile cube

Return to the 16 TB opening tensor. For every live depth-line pair, the center
calculation produces only three useful items:

```text
depth row, target wavelength column, opacity value
```

Many pairs can target the same pixel. Mathematically,

\[
K_{d,w}
=\sum_{l:j_l=w} k_{d,l}.
\]

On the Torch route, `_scatter_add_rows` performs this segmented sum without a
line axis in the output. Explain a write race with a two-line example: if two
workers read the same old pixel and both overwrite it, one contribution can
disappear. Scatter/index-add primitives guarantee that overlapping
contributions are combined according to the backend's reduction semantics.
“No lost update” does not promise the same floating-point association on every
device.

The teaching implementation must first construct a tiny dense `(D,L,W)`
oracle, sum over `L`, and assert agreement with the sparse center deposit. Then
delete the dense line axis from the scalable path. This is a main-text proof of
the algorithm, not an exercise.

### 5.9 Wings need reach estimates and two accumulation shapes

Do not rederive the Harris/Voigt profile. The new question is how far each
live depth-line pair must walk before its wing becomes negligible.

The ordinary wing reference is the source expression

```text
maximum(
    continuum_opacity_at_wing_anchor * LINE_CENTER_CUTOFF_RATIO,
    continuum_opacity_at_wing_anchor * WING_CUTOFF_FLOOR_RATIO,
)
```

with exact constants

```text
LINE_CENTER_CUTOFF_RATIO = 1e-3
WING_CUTOFF_FLOOR_RATIO = 1e-8
MAX_WING_PROFILE_STEPS = 1_000_000
```

Retain the two named policies in the narrative even though their current
shared continuum anchor makes the \(10^{-3}\) term larger. They describe
separate center and safety-floor roles.

The batched reach calculation predicts how many pixels remain above the
cutoff for each depth-line pair:

- lines with maximum reach at most
  `NARROW_WING_MAX_REACH = 128` use the batched narrow walker;
- narrow work is grouped into exact reach tiers `(1, 8, 32, 128)`;
- wider lines use the explicit `_wing_walk_metal` path;
- a line center outside the grid may still have a reachable wing anchor;
- every outward walk stops at the first below-cutoff point according to its
  branch's contiguous-walk rule or at a bounded grid/step limit.

Show one ordinary line whose weak wing uses a narrow tier and one strong line
that enters the wide path. Plot their predicted reach in pixels against
continuum-relative strength in a single panel. Then assert batched-versus-loop
deposit agreement on the compact fixture within the declared float32
accumulation tolerance.

### 5.10 Chunking bounds memory; reduction order defines a numerical route

A **chunk** is a contiguous or indexed subset of lines processed together so
temporary arrays do not scale with the full catalog.

The two production lanes solve the memory/race problem differently:

| lane | parallel work | private/shared storage | reduction order |
| --- | --- | --- | --- |
| synthesis Torch | metal index chunks of `METAL_CHUNK = 40_000`; narrow deposits use scatter/index add, wide lines walk explicitly | each call returns a `(D,W)` float32 chunk slab | pipeline adds chunk slabs sequentially in chunk order |
| atmosphere compact selector/deposit | one contiguous line chunk per Numba thread | private float32 `(chunk,D,W)` buffers | compiled fixed `for c in range(chunk_count)` sum |
| atmosphere detailed transitions | contiguous line chunks, each re-seeding the memoryless monotonic wavelength walk | private float32 buffers | fixed chunk-order sum; deterministic regrouping relative to the serial route |

For the atmosphere compact route, show the core pattern in a bite-sized
teaching kernel:

```text
parallel over chunks with prange
    accumulate this line interval into buffer[chunk]
serial over chunk number
    total += buffer[chunk]
```

The private buffer removes write races. The final serial loop fixes the chunk
reduction order. It does not reproduce the serial line-by-line association
exactly; it produces a deterministic float32 regrouping that must satisfy the
spectrum-level tolerance.

For synthesis, changing `metal_chunk` changes addition grouping and temporary
size, not the intended physics. Check at least two chunk widths against an
unchunked small oracle. Report maximum absolute and relative differences;
never conceal expected last-bit float32 differences behind “identical.”

Movement 7A closes with one professional single-panel plot of ordinary atomic
opacity versus wavelength at one labeled depth. It should show the checked Fe
I line embedded in an overlapping forest and annotate only the strongest
center. The scientific point is overlap: the final slab is a sum, not a list
of isolated profiles.

## 6. Movement 7B — Route the profiles that are not ordinary

Open this movement with three normalized shapes at one controlled depth:
ordinary symmetric line, broad hydrogen line, and asymmetric autoionizing
line. Use a conceptual schematic, not a quantitative multi-curve plot, so the
subsequent figures can each make one measured claim.

### 6.1 Hydrogen and deuterium are series, not isolated Voigt lines

Introduce the principal quantum numbers `n_lower` and `n_upper` as labels for
hydrogen energy levels. Closely spaced fine-structure components share the
catalog transition strength but are offset in frequency and combined with
tabulated weights. The current synthesis route computes those components for
each resolved `(n_lower, n_upper)` pair.

Hydrogen broadening depends on more state than the ordinary three-term Voigt
line. The exact `accumulate_hydrogen` state includes:

- `temperature`;
- `electron_density`;
- `mass_density`;
- `hc_over_kt`;
- `helium_neutral_population`;
- `molecular_hydrogen_population`;
- `hydrogen_partition_normalized_ion_stage_populations`;
- `microturbulence`;
- `hydrogen_neutral_partition_normalized_population`;
- `hydrogen_fractional_doppler_width`;
- `continuum_opacity`.

Explain the causal roles rather than listing them without interpretation:
temperature and microturbulence set the Doppler core; electrons produce strong
Stark wings; neutral H, He, and H2 perturb the line; the local continuum sets
the stopping threshold.

The hydrogen engine uses fine-structure components, an impact/quasi-static
Stark construction, radiative/resonance/van der Waals terms, neighboring-line
dominance cutoffs, and a conservative outward reach. It deposits only the
contiguous region that can remain relevant. Deuterium records share this
current route after their source wavelength and isotope classification are
fixed by the catalog.

State the current support boundary exactly. Synthesis routes neutral H/D
records with `n_lower >= 2`. A resolved record with `n_lower < 2` raises
`NotImplementedError`; Lyman support is not silently approximated by the
Balmer-oriented engine.

This limit is lane-specific. The atmosphere detailed type-`-1` route validates
principal levels in `1..100`, requires a nonzero in-range
`hydrogen_continuum_selector_index`, and builds
`HydrogenLineProfileEvaluator` state from H, H\(^+\), He, H2, electron, and
profile-table inputs. Its evaluator includes the dedicated Lyman-\(\alpha\)
resonance and quasi-molecular cutoff terms. The chapter must therefore say
“Lyman is guarded out of the synthesis H/D engine,” not “the complete
implementation has no Lyman profile.”

The first hydrogen quantitative figure should have one claim: increasing
electron density broadens the wings of the same transition at fixed
temperature and populations. Use one panel, the same line color with
light-to-dark density encoding, and a logarithmic opacity axis only if zero
handling is stated.

### 6.2 High series members dissolve into a pseudo-continuum

As the upper quantum number increases, neighboring hydrogen lines approach a
series limit and their Stark-broadened wings overlap. Treating them as forever
separate would leave an artificial comb where the physical levels are
dissolved by the plasma.

The synthesis catalog marks merged-continuum records with `n_upper == 99`.
The per-depth Inglis–Teller construction is

\[
n_{\rm IT}
=\frac{1600}{n_e^{2/15}},
\qquad
n_{\rm merge}=\max(n_{\rm IT}-1.5,1),
\]

\[
\texttt{merge\_wavenumber\_by\_depth}
=\frac{109737.312}{n_{\rm merge}^2}
\quad [{\rm cm}^{-1}].
\]

This is why `merge_wavenumber_by_depth` must be recomputed for each
atmosphere, even when the remaining `HydrogenInvariants` are cached. The
source builds each synthesis merged strength as
`oscillator_strength * 2 * n_lower**2`, sets
`last_resolved_upper_level = 81`, and deposits a plateau followed by a linear
tail between depth-dependent limits. The
merged records add pseudo-continuum opacity up to those limits, while
resolved lines are tapered or stopped by merge and neighboring-line
boundaries.

Use one single-panel plot of `merge_level` against `electron_density`. Its one
claim is that denser plasma lowers the last distinguishable level. A separate
original schematic should show discrete high-\(n\) levels crowding and fading
into a continuum edge; do not make one plot carry both ideas.

The atmosphere detailed route expresses its merged-continuum records
differently: a nonnormal negative `line_type` carries the last-level value, and
the kernel deposits the associated continuum over at most 1001 samples. Its
effective charge is 2 only for packed species slot 4 and 1 otherwise.
State this lane-specific encoding without forcing it into synthesis's
`n_upper == 99` convention.

### 6.3 Helium retains family-specific scaling and optional merge tapers

The synthesis route recognizes He I (`-3`), He3 I (`-4`), and He II (`-6`).
It reuses the already checked population, excitation, damping, and cutoff
logic, then evaluates the helium-specific wing walk.

For He3, the exact current scale is

```text
HELIUM3_ISOTOPE_SCALE = 1.155
effective_amplitude     = line_amplitude / 1.155
effective_doppler_width = doppler_width * 1.155
effective_damping_ratio = damping_ratio / 1.155
```

Do not paraphrase this as a generic mass law; present it as the implemented
source contract and then interpret the broader, lower-amplitude core.

The helium branch accepts per-depth `helium_core_weight_grid` and
`helium_tail_weight_grid` merge tapers. The packaged optical catalog currently
does not carry that taper metadata, so the standard pipeline passes `None` and
the branch uses zero grids. The chapter must show both the capability and the
actual standard-route status.

A compact check should verify the three He3 scaling identities directly and
show that `None` produces the same untapered result as explicit zero taper
grids.

### 6.4 Autoionizing resonances are asymmetric

Type-1 records represent resonances coupled to a continuum. Their current
profile is a Shore–Fano-like asymmetric branch rather than the ordinary
symmetric Harris/Voigt profile.

The branch reconstructs the required source meanings from the preserved raw
damping fields. A nonzero raw radiative or van der Waals log is exponentiated
directly; otherwise the ordinary normalized coefficient is multiplied back by
`12.5664 * frequency`. For the asymmetric Stark field, a positive raw log
becomes `-10**(-raw_log)`, a negative raw log becomes `10**raw_log`, and zero
falls back to the de-normalized ordinary coefficient. Its reduced-frequency
profile ratio has the implemented form

\[
\frac{q\,x+w}{x^2+1}\,\frac{1}{w},
\qquad
x=\frac{2(\nu-\nu_l)}{\gamma_{\rm rad}},
\]

where the source Stark field supplies the asymmetry \(q\) and the source van
der Waals field supplies the baseline/width-like value \(w\). Avoid presenting
those reused source columns as ordinary Stark and neutral-collision damping in
this branch.

The center amplitude is population- and excitation-dependent and must exceed
the same continuum-relative \(10^{-3}\) criterion. Red and blue walks proceed
separately. The synthesis route deposits only positive, above-cutoff
contiguous contributions and evaluates 1024-sample walk blocks until the
cutoff or grid edge. The atmosphere detailed route bounds each red and blue
walk to 2001 samples, adds the current sample, and then stops when that sample
is below its continuum threshold; it has no separate positivity mask. These
are different implementations of the same current source-specific profile
contract, not bitwise-equal intermediate algorithms.

One single-panel figure should vary the asymmetry parameter while holding the
center scale fixed. Its sole claim is that the two wings respond differently.

The coupled call behavior must be tested explicitly:

```text
do_metal=False -> neither ordinary nor autoionizing deposit
do_metal=True  -> ordinary and autoionizing branches are both eligible
```

Do not “clean up” that behavior into a new public flag in the textbook.

### 6.5 COR and PRD labels remain honest

Close the route discussion with two negative results:

- type-2 `COR` records are decoded and retained in catalog provenance but do
  not enter the standard synthesis invariant masks; the atmosphere detailed
  kernel also skips type 2;
- type-3 `PRD` records use the ordinary LTE type-0 profile route in both
  current lanes. No frequency-redistributing source function is computed.

These are not omissions to hide. They teach the reader to distinguish
“present in a source catalog” from “wired into the forward model.”

### 6.6 All branches add to one slab

End the construction with the actual additive structure:

```text
ordinary type 0/3 + autoionizing type 1
    + helium -3/-4/-6
    + hydrogen/deuterium -1/-2 and merged-series opacity
    = atomic line_mass_absorption_coefficient
```

In the synthesis pipeline, the shared slab is initialized as float32.
Stimulated emission is applied once inside each completed branch result before
that result is added. The compact final accounting cell should compute each
branch separately on the staged fixture, add them in pipeline order, and
assert agreement with the one-call/pipeline atomic slab within the declared
float32 tolerance.

No molecular line contribution is included in this checkpoint.

## 7. Visible code-cell ledger

The notebook should target 17 substantial visible cells. Short display-only
cells do not count, but large hidden blocks are not an acceptable way to evade
the density gate. Each cell should normally remain 10–30 lines, with 60 as a
soft maximum and 80 as a hard maximum.

| cell | reader action | immediate check or visible result |
| ---: | --- | --- |
| 1 | load the self-contained atomic teaching fixture and recover the Chapter 6 Fe I row | provenance and field names, no source tour |
| 2 | apply isotope, energy-shift, wavelength, and damping corrections | raw/corrected audit table |
| 3 | build requested and 16-sample context grids | bitwise interior equality |
| 4 | compute exact line-size margins | inclusive blue/red boundary checks |
| 5 | assemble `LineCatalog` and route masks | route-count table; catalog order retained |
| 6 | separate window invariants from atmosphere state | rebuild/cached-value equality on compact fixture |
| 7 | compute the atmosphere depth-envelope and exact scalar keep clauses | three isolated inequality-boundary records |
| 8 | compare Python, serial `njit`, and parallel `prange` selectors | identical masks; cold and warm timings separated |
| 9 | compare synthesis live center masks with the atmosphere's eight-layer depth gate | lane-specific selection table and exact count meaning |
| 10 | compare dense center oracle with sparse scatter-add | array equality/tolerance and memory estimate |
| 11 | estimate wing reach and route narrow versus wide lines | tier counts and bounded reach |
| 12 | compare batched wing deposit with explicit loop | max absolute/relative difference |
| 13 | compare chunk widths and private-buffer reduction with small serial oracle | no lost updates; reported float32 regrouping |
| 14 | build and plot the ordinary atomic forest at one depth | Movement 7A checkpoint |
| 15 | evaluate hydrogen fine structure/Stark width and series merging | density trend plus explicit Lyman guard |
| 16 | evaluate He/He3 and autoionizing controlled cases | isotope identities, taper-zero identity, asymmetry |
| 17 | route every fixture family into the final additive atomic slab | branch trace, COR/PRD status, final parity check |

If these cells become too large, factor reusable helpers into
`book/chapters/chapter07_atomic_lines.py` with descriptive names. The notebook
must still show the decisive equations, masks, and reductions in bite-sized
form. Do not paste production functions into Markdown or hide the causal step
behind a single opaque `run_chapter07()` call.

## 8. Figure and schematic contract

### 8.1 Quantitative figures

Every quantitative figure should be one panel and make one claim.

1. **Catalog funnel:** number of records after decode, window keep, route
   eligibility, and live depth-line keep. A horizontal count or log-count plot;
   do not mix it with timings.
2. **Wing reach:** predicted reach in grid pixels versus
   continuum-relative line strength, marking the 128-pixel narrow/wide
   boundary.
3. **Ordinary forest:** atomic opacity versus wavelength at one named depth,
   with the Chapter 6 line identified once.
4. **Hydrogen density effect:** one transition at fixed temperature for a
   small electron-density sequence.
5. **Series dissolution:** `merge_level` versus electron density.
6. **Autoionizing asymmetry:** controlled asymmetry values at fixed center
   scale.

Do not combine these into paper-style multi-panel summaries. The paper may
inspire typography and restraint, but the textbook must walk through one
relationship at a time.

Plot requirements:

- consistent professional font family and sizes across the book;
- colorblind-safe palette with physical meaning assigned consistently;
- wavelength and opacity axes labeled with units;
- no unexplained scientific-notation offset text;
- light major grid only when it helps read values;
- direct annotation preferred to a legend for one highlighted line;
- no chart junk, decorative gradients, or unlabeled secondary axes;
- deterministic figure data and saved asset names;
- `tight_layout`/constrained layout and clipping checks in both notebook and
  local reader.

### 8.2 Original conceptual schematics

Generate new schematics using a chapter-local Python script that adopts the
official website generator's clean visual grammar—warm dark-blue ink,
off-white background, restrained cyan/gold/coral accents, rounded scientific
forms, and generous whitespace—without reusing website images or
compositions.

During implementation, inspect the `.py` schematic generators in
`~/payne-zero-website` and adapt their reusable typography, palette, line, and
export primitives into the local chapter script. Record that provenance, but
do not import from the website repository at notebook runtime. Every scene
layout and scientific composition below must be new.

Required original compositions:

1. **The catalog funnel.** A few readable source cards enter correction,
   window, and route gates; only a sparse set of colored records reaches the
   opacity grid. No fake numerical labels.
2. **Sparse deposition.** Several depth-line pairs send arrows to shared
   wavelength pixels in a `(D,W)` slab; overlapping arrows visibly add rather
   than overwrite.
3. **Ordinary and special route tree.** One source catalog separates into
   ordinary/PRD, autoionizing, H/D, He, and COR-unwired branches, then eligible
   branches rejoin one additive slab.
4. **Hydrogen series merging.** Low levels remain discrete; high-\(n\) levels
   crowd, broaden, and dissolve smoothly into a continuum edge.
5. **Two safe parallel reductions.** Left: Torch scatter-add into wavelength
   targets. Right: CPU `prange` chunks write private slabs and reduce in fixed
   order. The image teaches different race-avoidance strategies, not hardware
   branding.

Each schematic must include an adjacent prose paragraph that explains what to
notice and what is intentionally omitted. Alt text should state the causal
relationship rather than list colors. Generated labels must be audited for
spelling and mathematical legibility.

## 9. Self-contained data and source-staging requirements

Implementation must never import from `~/payne-zero` or read a user's external
catalog tree at notebook runtime. The pinned source is read-only authority; the
textbook receives the minimal files and compact data it needs.

### 9.1 Source fingerprints audited for this contract

| pinned source file | SHA-256 |
| --- | --- |
| `payne_zero_synthesis/atomic_lines.py` | `0fa52833fb16487da1d5bfaaf5628a46751f888c1a57894a5037daa6d6667ab0` |
| `payne_zero_synthesis/line_opacity.py` | `639b95c3812f1a7d227b797fa89a4d6ef9725d5f0e1284f3d49cf86844278275` |
| `payne_zero_synthesis/hydrogen_lines.py` | `81ab3ee2ca9ecd1994ddde8f01e09535c5b74f7beec5afe98a3c63b44677dcca` |
| `payne_zero_synthesis/pipeline.py` | `465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22` |
| `payne_zero_atmosphere/line_selection.py` | `b2c62fdf5e1fe43f33022184bfeff88985b13331354e3c745c7dab3a6b634fef` |
| `payne_zero_atmosphere/line_catalog.py` | `2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92` |
| `payne_zero_atmosphere/line_opacity.py` | `d0f9c43919be58a42547e12b7abc22161a7558bf17abbcd375ab04ccf57d7cc6` |

At contract time, the textbook already stages synthesis `atomic_lines.py`,
`line_opacity.py`, and `pipeline.py`, plus atmosphere `line_catalog.py` and
`line_opacity.py`. It does not yet stage synthesis `hydrogen_lines.py` or
atmosphere `line_selection.py`. A later implementation pass must copy the
pinned files into the textbook's `src` tree, update the manifest, and verify
that imports remain local. This contract does not perform that copy.

### 9.2 Minimal chapter data

Add one compact, human-auditable fixture containing:

- the Chapter 6 ordinary Fe I record;
- a corrected isotope/energy-shift ordinary record;
- a line at each exact margin boundary;
- at least one type 1, type 2, and type 3 record;
- H I, D I, He I, He3 I, and He II records;
- a resolved Balmer-family line and an `n_upper == 99` merged record;
- a deliberately unsupported Lyman record used only to show the guard;
- enough overlapping ordinary lines to exercise shared center pixels, narrow
  wings, wide wings, and chunk boundaries.

The fixture must preserve original record order and record-level provenance.
Store only the fields needed to reconstruct the route and opacity checks; do
not copy a multi-gigabyte catalog into the repository.

Reuse existing local Chapter 3–6 atmosphere, continuum, line-profile, and
exponential tables where their manifest identities match. Add a compact
atmosphere packed-word selection fixture that straddles the three exact keep
inequalities. Record every new file, shape, dtype, unit, origin, and checksum
in `data/MANIFEST.json` and `data/README.md`.

The optional full-catalog path may be documented after the self-contained
chapter works, but it cannot be required for the main notebook or tests.

## 10. Main-text checks and acceptance gates

Checks appear immediately after the idea they validate. A final compact audit
may summarize them, but it must not be the first time the reader sees a
failure.

### 10.1 Catalog and grid

- corrected hand-selected records match the pinned source transformation;
- `loggf`, wavelength, species/stage, defaults, normalized damping, and raw
  damping provenance are checked independently;
- catalog order and route counts survive cache write/reload;
- cache identity changes when window, resolution, source identity, sort, or
  isotope-correction policy changes;
- requested wavelength samples are bitwise unchanged inside the 16-sample
  context;
- center and wing-anchor indices match the host-float64 scalar reference;
- exact blue/red margin boundary records remain selected.

### 10.2 Selection and acceleration

- scalar and vectorized depth-envelope values agree;
- the strict population floor and two inclusive center clauses are tested
  separately;
- Python, serial `njit`, and parallel `prange` masks are byte-identical;
- cold compilation time is separated from warm execution time;
- `fastmath` remains disabled;
- the NumPy-pretabulated exponential path matches the source decision mask;
- selected packed rows decode to an internally consistent
  `SelectedLineCatalog`.

### 10.3 Sparse ordinary deposits

- the tiny dense `(D,L,W)` oracle equals the sparse center sum;
- two lines targeting one pixel both contribute;
- batched narrow-wing and explicit loop deposits agree within the documented
  float32 tolerance;
- an off-grid center with a reachable wing contributes inside the context
  grid;
- at least one line crosses the 128-pixel narrow/wide routing boundary;
- two synthesis chunk widths and serial/parallel atmosphere reductions report
  their actual maximum errors;
- the ordinary forest is finite, nonnegative, and additive.

### 10.4 Special routes

- the route table accounts for every fixture record exactly once as deposited,
  deliberately skipped, or unsupported;
- `do_metal=True` branch tracing includes autoionizing opacity;
- type 2 contributes exactly zero in the standard routes;
- type 3 matches the ordinary type-0 profile for otherwise identical inputs;
- the autoionizing profile changes red/blue balance when its asymmetry field
  changes; the synthesis route remains nonnegative while the atmosphere route
  matches its scalar add-then-stop reference;
- hydrogen fine-structure component weights match the source-table
  normalization for the selected transition;
- increasing electron density moves the Inglis–Teller merge level in the
  expected direction;
- resolved `n_lower < 2` synthesis input triggers the explicit Lyman guard;
- merged-series opacity joins the resolved region without an unexplained
  bookkeeping gap in the controlled fixture;
- He3 amplitude, width, and damping scaling equal the exact 1.155 identities;
- `None` and zero helium taper grids agree for the packaged-metadata case.

### 10.5 Integration

- separately accumulated ordinary/autoionizing, He, and H/D/merged slabs sum
  to the pipeline-order atomic slab;
- stimulation is applied exactly once in the synthesis route;
- the atmosphere and synthesis outputs are validated against their own
  lane-specific pinned authorities, not against one another as identical
  intermediates;
- with identical staged inputs, the final Chapter 7 atomic slab satisfies the
  predeclared Payne Zero parity tolerances and route counts;
- no molecular line opacity and no radiative-transfer output appears.

Every tolerance must be declared beside the quantity and dtype before the
comparison runs. Bitwise identity is required for integer routes, catalog
order, packed selected words, discrete indices, and context-grid interior.
Floating accumulation comparisons use measured absolute/relative tolerances
declared for the active branch dtype, including the synthesis hydrogen
CPU/CUDA high-precision route.

## 11. Redundancy and deferral audit

The chapter must not:

- rederive the ordinary line-strength, FASTEX, Doppler, damping, or
  Harris/Voigt equations from Chapter 6;
- rederive Saha populations, molecular equilibrium, or continuum opacity;
- teach source files in directory order;
- paste production functions into prose;
- call type 3 a working PRD transfer method;
- imply that type 2 COR contributes opacity;
- hide the unsupported Lyman boundary;
- conflate `SelectedLineCatalog` with `LineTransitionCatalog`;
- conflate window keep, atmosphere source keep, and live depth-line keep;
- call scatter-add bitwise deterministic across every backend;
- introduce molecular family corrections before Chapter 8;
- compute an emergent flux before Chapter 9;
- turn device placement, fusion, and end-to-end GPU performance into a second
  Chapter 10 inside this chapter;
- add end-of-chapter exercises.

The chapter may say once that Torch scatter primitives can run on an
accelerator and may measure the CPU/GPU difference if the current machine
supports both. The full GPU synthesis architecture, transfer minimization, and
performance ladder remain Chapter 10.

## 12. Chapter summary and causal handoff

End with a short reader-facing summary, not an implementation changelog:

1. A source line becomes usable only after its corrections, wavelength,
   damping provenance, and route have been made explicit.
2. Conservative window and atmosphere selection remove impossible work;
   local depth-line cutoffs remove work that is negligible in the current
   state.
3. Sparse center/wing deposits replace an impossible dense `(D,L,W)` tensor.
4. Scatter-add and private-buffer reductions prevent lost overlapping
   contributions, while chunk order makes floating-point grouping explicit.
5. Ordinary and PRD-tagged records share the current LTE route; autoionizing,
   hydrogen/deuterium, helium, and merged-series records need special paths;
   COR remains parsed but unwired.
6. The result is one checked atomic line-opacity slab, not yet an emergent
   spectrum.

The final link to Chapter 8 should be causal:

> Our atomic records now share a trustworthy route into one opacity slab. Cool
> stellar spectra add molecular bands, but their sources do not share one
> encoding or one correction rule. Chapter 8 will compile those heterogeneous
> molecular sources into the same kind of sparse, checked contribution without
> pretending that every available family is wired into the runtime.

That sentence is the only forward reference required in the close.
