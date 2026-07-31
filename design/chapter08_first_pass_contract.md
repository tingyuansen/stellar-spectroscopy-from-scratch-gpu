# Chapter 8 first-pass contract — Molecular Bands and Source Compilation

Status: bounded reader-facing design; no implementation or publication authority  
Pinned Payne Zero commit: `9c44001feae40b85146630499e6f8a5fed42e5af`  
Audience: final-year undergraduate / first-year graduate student  
Canonical title: **Molecular Bands and Source Compilation**

## 0. Canonical placement and non-negotiable boundaries

This chapter follows the canonical fifteen-chapter architecture in `PLAN.md`
and `design/global_chapter_contracts.md`:

```text
Chapter 4: solve molecular equilibrium and publish molecular populations
Chapter 6: build one ordinary line profile
Chapter 7: select and sparsely deposit atomic forests and special profiles
                     |
                     v
Chapter 8: translate molecular source families into checked line opacity
                     |
                     v
Chapter 9: solve radiative transfer through the completed opacity
```

Chapter 8 owns the molecular **source-to-opacity** problem. It must cover both
production lanes without pretending they share one input format:

1. the atmosphere lane reads converted diatomic, TiO, water, and optional
   H\(_3^+\) source arrays, applies family-specific corrections or decoding,
   and passes admitted records through the common `SelectedLineCatalog`
   machinery built in Chapter 7;
2. the synthesis lane compiles manifest-ordered text bands and packed TiO into
   a common molecular structure of arrays, maps each species to the molecular
   population lane built in Chapter 4, and performs chunked Torch deposition.
   Its H\(_2\)O compiler is real and must be verified, but the standard
   synthesis pipeline does not call it.

The chapter does **not**:

- solve molecular equilibrium again;
- rederive the ordinary LTE strength, Doppler, damping, Harris/Voigt, cutoff,
  or scatter-add mathematics from Chapters 6–7;
- assemble the complete synthesis pipeline or teach invariant-cache residence
  across many stars, which belongs to Chapter 10;
- teach the atmosphere iteration's select-once/reuse lifecycle, which belongs
  to Chapter 11;
- compute emergent intensity or flux, which belongs to Chapter 9;
- introduce fitting, line calibration, NLTE, `torch.compile`, or an invented
  GPU atmosphere path.

The chapter has two visible movements:

1. **8A — A common record from unlike molecular sources:** molecular band
   intuition; text, TiO, H\(_2\)O, and atmosphere encodings; scalar and serial
   cached-Numba compilation; exact availability/wiring matrix; provenance and
   cache identity.
2. **8B — From molecular populations to a line-opacity slab:** exact population
   mapping; molecular masses and Doppler widths; device invariants; live
   cutoffs; bounded line/pair chunks; center, near-wing, and far-wing deposits;
   one stimulated-emission application; component and combined checks.

Target 17 substantial visible code cells and never exceed 18 in the first
pass. Prefer two clearly announced movements and web anchors over another
chapter. If the implemented material cannot fit 18 bite-sized cells without
compressing definitions or omitting immediate interpretation, the density
finding must be escalated to the global plan rather than hidden in long cells.

There are no detached exercises. Every prediction, limiting case, timing, and
debugging check belongs in the causal main text at the point where it resolves
a live question.

## 1. The chapter's single question

Open with a cool-atmosphere line-opacity window after Chapter 7's atomic
contribution has been calculated. The atomic forest is real, but broad combs
of missing absorption remain when the molecular contribution is omitted. Do
not begin with filenames, compiler functions, or a feature table.

Ask:

> One molecule supplies a population, yet its rotations, vibrations, and
> electronic states can create millions of transitions stored in several
> incompatible formats. How can those sources become one checked opacity
> contribution without losing isotope weights, record order, population
> ownership, or the distinction between code that exists and physics that is
> actually wired into the runtime?

The destination is:

```text
Chapter 4 molecular populations
                 +
text bands | packed TiO | packed H2O | atmosphere converted families
                 |
family-specific decoding and correction on CPU
                 |
common line fields and exact species mapping
                 |
host-built, device-resident invariants
                 |
bounded sparse center and wing deposits
                 |
checked molecular line_mass_absorption_coefficient
```

The opening should make the scale concrete without turning into a benchmark.
At the pinned source identity:

- the synthesis text manifest contains 32 bands and 22,377,706 records;
- packed TiO contains 37,744,499 records;
- packed H\(_2\)O contains 65,912,356 records;
- the atmosphere's converted diatomic source contains 12,488,322 records.

These counts motivate compilation, memory mapping, and chunking. They do not
justify copying the multi-gigabyte source tree into the normal book.

Payne Zero should be named at exact source interfaces, default-feature
boundaries, and parity checks. The surrounding narrative should remain about
the physics and algorithm the reader is building.

## 2. Reader promise and prerequisite floor

By the end of Chapter 8, the reader should be able to explain and reproduce:

- why molecular spectra form bands rather than one isolated line per
  molecule;
- how one total molecular population becomes many lower-state line
  populations without applying the molecular partition function twice;
- why a source's existence, a compiler's existence, and a standard runtime
  path are three different claims;
- the different meanings of the synthesis text, TiO, and H\(_2\)O encodings;
- why the atmosphere diatomic, TiO, and water families require different
  corrections before sharing Chapter 7's selected-record path;
- why the standard atmosphere route includes water but the standard synthesis
  route does not;
- why a mass-table entry for H\(_3^+\) is metadata rather than runtime wiring;
- why manifest order and record order are part of reproducibility;
- how a readable scalar text compiler becomes a serial
  `@numba.njit(cache=True)` compiler without changing the emitted order;
- why the compiler does not use `prange`, even though the atmosphere's
  independent keep tests do;
- how `species_code // 6 - 1` finds the exact stage-5 population column;
- how molecular mass, temperature, and microturbulence set the fractional
  Doppler width;
- why source compilation happens on the host while the large opacity algebra
  and sparse deposits belong on the selected Torch device;
- how `CHUNK_LINES`, `PAIR_CHUNK`, and offset blocks bound different temporary
  array shapes;
- how a cache can accelerate compilation without becoming a scientific input;
- which exact checked product Chapter 9 receives.

Assume only the prerequisite floor in `BIBLE.md`. Define “molecule,”
“isotopologue,” “rotational state,” “vibrational state,” “band,” “manifest,”
“packed field,” “memory mapping,” “compiler,” “cache,” and “chunk” when they
first appear. Do not assume prior molecular spectroscopy, Numba, Torch, or GPU
experience.

## 3. Exact reads, writes, notation, and lane separation

Use \(D\) for depth, \(W\) for wavelength samples, \(L\) for compiled lines,
and \(M\) for the unique molecular species represented in one window. These
symbols explain shapes; reusable code keeps the exact implementation names.

### 3.1 Physical state read from earlier chapters

Chapter 8 consumes, but does not recompute, the following state:

| physical meaning | exact name | shape | unit | owner |
| --- | --- | --- | --- | --- |
| partition-normalized public population cube | `partition_normalized_populations` | `(D,6,139)` | cm\(^{-3}\) | Chs. 3–4 |
| molecular line-population lane | `partition_normalized_populations[:, 5, :]` | `(D,139)` | cm\(^{-3}\) | Ch. 4 |
| mass density | `mass_density` | `(D,)` | g cm\(^{-3}\) | Ch. 3 |
| electron density | `electron_density` | `(D,)` | cm\(^{-3}\) | Ch. 3 |
| temperature | `temperature` | `(D,)` | K | Ch. 1/3 |
| microturbulent speed | `microturbulence` | scalar or `(D,)` | cm s\(^{-1}\) | Chs. 2–3 |
| excitation factor input | `hc_over_kt` | `(D,)` | cm | Chs. 3/6 |
| neutral-collision proxy | `collision_density_proxy` | `(D,)` | cm\(^{-3}\) | Chs. 3/6 |
| continuum comparison opacity | `continuum_opacity` | `(D,W)` | cm\(^2\) g\(^{-1}\) | Ch. 5 |
| synthesis wavelength grid | `wavelength_grid` | `(W,)` | nm | Ch. 7 |

The stage-5 population is already divided by the molecular partition function.
The line kernel multiplies it by the lower-state Boltzmann factor
\(\exp(-\tilde E_l hc/kT)\). It must not divide by a molecular partition
function again.

The atmosphere lane consumes the packed
`partition_normalized_population_over_mass_density_and_fractional_doppler_width`
state established in Chapters 3, 6, and 7. Chapter 8 does not derive that
packed layout again.

### 3.2 Synthesis compiler output

Every synthesis source compiler returns the same eight per-line arrays:

| exact key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `center_index_1based` | `(L,)` | `int32` | center on the compiler's geometric grid, one based |
| `classical_line_strength` | `(L,)` | `float32` | Chapter 6 integrated-strength coefficient |
| `species_code` | `(L,)` | `int16` | exact molecular population code |
| `lower_excitation_cm` | `(L,)` | `float32` | lower-state wavenumber energy |
| `radiative_damping` | `(L,)` | `float32` | normalized radiative coefficient |
| `stark_damping` | `(L,)` | `float32` | normalized electron coefficient |
| `van_der_waals_damping` | `(L,)` | `float32` | normalized neutral-collision coefficient |
| `margin_class` | `(L,)` | `int16` | compiler record class; currently `7` for these molecular outputs |

All eight arrays must have identical length and preserve the concatenation
order. `margin_class` is part of the compiled-source output but is not a field
of `MolecularLineCatalog`; the runtime catalog consumes the seven physical
line arrays and rebuilds its derived indices.

The combined persistent compiler product also carries scalar
`log_grid_ratio` and `grid_origin_index`. Here the exact internal argument and
field name is `resolution`; explanatory mathematics may call the physical
quantity \(R_{\rm grid}\). It is not instrumental resolving power.

### 3.3 Runtime catalog and invariant products

The visible `MolecularLineCatalog` contract must include every exact field:

| field group | exact fields | role |
| --- | --- | --- |
| compiled physical fields | `center_index_1based`, `classical_line_strength`, `species_code`, `lower_excitation_cm`, `radiative_damping`, `stark_damping`, `van_der_waals_damping` | source-faithful line data |
| derived line indices | `center_index`, `species_population_column`, `wavelength_nm` | zero-based target, public population column, reconstructed line center |
| grid provenance | `log_grid_ratio`, `grid_origin_index` | reconstruct centers without inventing a new grid |
| compact species inventory | `unique_species_codes` | one row per molecular species in the window |

The device-resident `MolecularLineInvariants` contract must likewise be
visible:

`wavelength_grid`, `n_wavelengths`, `local_resolving_power`,
`classical_line_strength`, `lower_excitation_cm`, `radiative_damping`,
`stark_damping`, `van_der_waals_damping`, `center_index`,
`line_species_index`, `line_wavelength_nm`, `species_code`,
`species_population_column`, `species_mass_amu`,
`harris_profile_h0_table`, `harris_profile_h1_table`, and
`harris_profile_h2_table`.

Do not replace either source dataclass with a cleaner textbook-only API.
Teaching helpers may expose small local arrays, but they must converge to
these exact field names before a reusable boundary is presented.

### 3.4 Chapter outputs

| product | shape | unit/dtype | exact meaning |
| --- | --- | --- | --- |
| `molecular_opacity` | `(D,W)` | cm\(^2\) g\(^{-1}\), float32 | synthesis molecular line absorption after optional stimulation |
| `line_mass_absorption_coefficient` contribution | `(D,W)` | cm\(^2\) g\(^{-1}\), float32 | the same slab when added to the shared synthesis line accumulator |
| per-family atmosphere `SelectedLineCatalog` checkpoints | one-dimensional equal-length fields | packed source convention | records admitted by each family-specific selector before common deposit |
| atmosphere molecular-family opacity contributions | `(D,W)` | cm\(^2\) g\(^{-1}\), float32 | family-separated teaching checks through Chapter 7's common deposit |
| feature-status rows | small machine-readable table | categorical | compiler availability, default source availability, and runtime wiring kept separate |

The standard synthesis call uses `apply_stim=True`, so stimulated emission is
applied once after all molecular chunks have accumulated. The atmosphere
selected-line product remains in its own lane-specific convention. Never
compare the two intermediate slabs as if they must be byte-identical.

## 4. The physical bridge: one molecule, many transitions

### 4.1 Why molecular lines gather into bands

Before any format is introduced, build the physical reason a molecular
catalog is large. An atom's bound electron has discrete energies. A molecule
adds approximately separable rotational and vibrational motion:

\[
E \simeq E_{\rm electronic}
  + h\nu_{\rm vib}\left(v+\frac12\right)
  + B\,J(J+1).
\]

Define:

- \(v=0,1,2,\ldots\) as the vibrational quantum number;
- \(J=0,1,2,\ldots\) as the rotational quantum number;
- \(h\nu_{\rm vib}\) as the approximate spacing of a harmonic vibration;
- \(B\) as the rotational constant in energy units.

One electronic or vibrational change can therefore occur for many populated
rotational states. The closely spaced transition centers form branches and
bands. Real source lists supply the line centers, strengths, excitation
energies, isotope identities, and damping information; the runtime does not
solve molecular quantum mechanics from first principles.

Use one conceptual schematic and one compact code-generated stick spectrum to
make only this claim: **one species population can feed many individual
transitions**. Do not present the source list's millions of rows before this
need exists.

### 4.2 Population ownership

For molecular species code \(s\), the exact synthesis population column is

\[
p(s)=\left\lfloor\frac{s}{6}\right\rfloor-1.
\]

The line amplitude uses

\[
A_{d,l}
=S_l\,
\frac{(n_s/U_s)_d}{\rho_d\,(\Delta\nu_D/\nu)_{d,s}}\,
\exp\!\left(-\tilde E_l\frac{hc}{kT_d}\right),
\]

where \(S_l\) is the Chapter 6 line-strength coefficient. Present this as a
product in prose, not as an unexplained long variable. Define every term and
then point to
`species_population_doppler_ratio`. Small exact examples should include:

| molecule | `species_code` | `species_population_column` |
| --- | ---: | ---: |
| H\(_2\) | `240` | `39` |
| CO | `276` | `45` |
| TiO | `366` | `60` |
| H\(_2\)O | `534` | `88` |

These are source interfaces, not a newly invented periodic-table convention.
The Chapter 4 CO chain `276 -> 608 -> [depth,5,45]` may be recalled once in a
single sentence. Do not reopen the equilibrium catalog or Newton solve.

## 5. Movement 8A — Compile unlike molecular sources into common fields

### 5.1 Three synthesis source encodings, not one universal parser

Introduce the synthesis sources through the physical information they must
provide:

| family | stored representation | distinguishing information |
| --- | --- | --- |
| manifest text bands | one combined NPZ, eight arrays under each `<band>/...` prefix | source code and isotope dispatch; optional energy-derived wavelength; predicted-line sign convention; scaled damping |
| Schwenke TiO | 16-byte structured packed records | fine log-wavelength code, isotope/species code, lower-energy and strength table indices, three damping indices |
| Partridge–Schwenke H\(_2\)O | 8-byte structured packed records | fine log-wavelength code; the signs of energy and strength encode the isotopologue |

The normal book should show one or two records from each format. It must not
print a large file listing or paste a production parser into Markdown.

The combined text-band catalog contains eight arrays per band:

`stored_wavelength_nm`, `log_oscillator_strength`, `first_energy_cm`,
`second_energy_cm`, `source_code`, `isotope_index`,
`radiative_damping_log_scaled`, and `upper_label_is_ground_state`.

The pinned manifest order is scientific/provenance state. The standard
pipeline obtains

```text
band_names = [entry["band"] for entry in manifest["text_sources"]]
```

and compilation concatenates bands in precisely that order.

### 5.2 Read one text-band record before compiling a catalog

Use `_parse_text_line` only to teach the fixed-width source semantics that
produced the combined NPZ. State clearly that synthesis runtime reads the
converted NPZ and does not parse raw text.

For one record, build the exact causal order:

1. Reject a zero stored wavelength.
2. When `include_predicted_lines=False`, reject a record if either term energy
   is negative.
3. If `use_energy_level_wavelengths=True` and the absolute energy difference
   is positive, use

   \[
   \lambda_{\rm nm}
   =\frac{10^7}
          {\left||\tilde E_2|-|\tilde E_1|\right|};
   \]

   otherwise use `abs(stored_wavelength_nm)`.
4. Retain wavelengths from
   `start_wavelength_nm - 0.01` through
   `end_wavelength_nm + 0.1`. Under the energy-derived policy, also retain the
   exact source-file position guard extending 10 nm beyond those bounds.
5. Dispatch `(source_code, isotope_index)` to an exact `species_code` and two
   base-10 isotope log weights. Unknown dispatch pairs are skipped.
6. Form

   \[
   gf = 10^{\log(gf)_{\rm source}
       +\Delta_{\rm isotope,1}+\Delta_{\rm isotope,2}},
   \qquad
   \tilde E_l=\min(|\tilde E_1|,|\tilde E_2|).
   \]

7. Convert the center to `center_index_1based` on the geometric grid, form the
   Chapter 6 `classical_line_strength`, and normalize damping by the source's
   numerical `12.5664 * line_frequency_hz`.
8. Use the exact text-family default Stark and van der Waals values. If
   `upper_label_is_ground_state` is true, apply the smaller exact pair of
   defaults.

The production pipeline calls `compile_molecular_text` with
`use_energy_level_wavelengths=True` and leaves
`include_predicted_lines=False`. Show the public defaults separately from the
standard call; do not blur them.

Immediate checks:

- stored and energy-derived wavelength policies agree when the stored center
  was generated from the same positive energy difference;
- a zero difference falls back to the stored wavelength;
- a negative term energy is present only when predicted lines are enabled;
- isotope-log corrections multiply, rather than add to, linear \(gf\);
- an unknown dispatch code produces no compiled record;
- all eight output arrays have the exact dtype contract in Section 3.2.

### 5.3 The scalar compiler earns the serial cached-Numba compiler

First write a readable scalar loop over a tiny manifest-ordered subset. Its
result is the interpretation oracle. Then introduce
`_get_compiled_band_kernel()`:

```python
@numba.njit(cache=True)
def kernel(...):
    for row in range(stored_wavelength_nm.shape[0]):
        ...
```

Explain each decorator term:

- `njit` compiles numerical Python to machine code;
- `cache=True` preserves compiled machine code for a compatible later process;
- the absence of `parallel=True` and `prange` is deliberate.

The compiler emits a compacted ordered stream. A line that is rejected changes
the output position of every later accepted line. The pinned path therefore
uses a serial loop and exact scalar arithmetic order. Do not manufacture a
parallel compiler for pedagogical symmetry.

This is different from the atmosphere selector. Its keep decision is
independent per source row, so Chapter 7's common selector and Chapter 8's
special water selector use `@njit(cache=True, nogil=True, parallel=True)` and
`prange` to build masks. Output rows are gathered in source order after the
mask exists. Put the contrast in one small table:

| task | dependency | pinned parallel policy |
| --- | --- | --- |
| synthesis text compiler | ordered compact output | serial cached `njit` |
| atmosphere family keep mask | one independent Boolean per record | compiled `prange` |
| molecular opacity deposition | many depth-line/pixel operations | Torch device kernels and scatter-add |

Measure scalar Python, cold `njit`, and warm `njit` separately on the compact
subset. Timing is interpretation, not proof. Record-by-record exact equality
is the parity gate.

### 5.4 TiO: a packed log grid and isotope table

The TiO compiler must be taught as its own decoder, not forced through the
text parser.

The packed record fields are:

`wavelength_code`, `isotope_species_code`, `lower_energy_code`,
`log_oscillator_strength_code`, `radiative_damping_code`,
`stark_damping_code`, and `van_der_waals_damping_code`.

The source wavelength uses the fine logarithmic step

\[
\Delta\ln\lambda
=\ln\!\left(1+\frac{1}{2{,}000{,}000}\right).
\]

The 32768-entry packed lookup is

\[
T_i=10^{(i-16384)\,0.001},
\]

stored as float32. Decode the isotope index from the packed species field and
use the exact five TiO isotope fractions
`[0.0793, 0.0728, 0.7394, 0.0551, 0.0534]`. The synthesis species code is
`366`.

The compiler:

- memory maps canonical `.npy` input or legacy raw binary input;
- reconstructs vacuum wavelength, with optional vacuum-to-air conversion;
- keeps a one-nanometer margin on both sides;
- converts packed energy, strength, and damping indices through the lookup;
- includes the isotope fraction in `classical_line_strength`;
- returns the common eight-array contract.

The standard pipeline calls `compile_tio_schwenke(...,
use_vacuum_wavelengths=True)`. A compact boundary check should show that
toggling the public wavelength flag changes centers but not the decoded
isotope identity.

### 5.5 H\(_2\)O: signed fields encode four isotopologues

The H\(_2\)O record has:

`wavelength_code`, `signed_lower_energy_code`, and
`signed_log_oscillator_strength_code`.

The two signs define four isotopologue cases. The exact isotope fractions are
`[0.9976, 0.0004, 0.0020, 0.00001]`, and the synthesis species code is `534`.
Use a four-row sign table in the main text and decode it in one bite-sized
cell. Do not call the signs positive/negative energy physics; they are a
storage code.

The H\(_2\)O compiler uses the same fine log-wavelength step and packed
strength lookup as TiO, but it does not share the TiO record layout:

- lower excitation is `abs(signed_lower_energy_code)` in cm\(^{-1}\);
- strength uses `abs(signed_log_oscillator_strength_code)` and the decoded
  isotope fraction;
- the line-strength frequency is the reconstructed vacuum frequency;
- the optional air correction changes the synthesis center;
- radiative damping follows the exact wavelength-dependent source rule;
- the common output contract is returned.

Run `compile_h2o_partridge` on a compact teaching slice and verify it against
the pinned compiler. Then stop. Do not add its result to the standard synthesis
catalog, because the pinned standard `_compile_molecular` does not call this
function.

### 5.6 The atmosphere families have different converted encodings

Now turn to the atmosphere lane. The physical need is the same—family records
must reach the common Chapter 7 selected-record representation—but the source
adapters are different.

#### Diatomic

- `read_diatomic_line_catalog` yields `(N,4)` packed `int32` words.
- Canonical `.npy` data already have historical Fortran record markers
  removed; the legacy reader checks and strips 16-byte markers.
- `_diatomic_log_strength_offsets` maps the packed molecular/isotope code
  through the exact lookup table.
- `select_standard_line_words` applies those offsets and fixes the Stark
  packed field to `1`.
- Population slots remain source-record owned; this family uses the common
  Chapter 7 keep and deposit path.

#### TiO

- `read_standard_line_catalog` yields `(N,4)` packed `int32` words.
- `_titanium_oxide_log_strength_offsets` derives one of five exact isotope
  offsets from the packed species code.
- Selection uses `population_slot_override=895`,
  `stark_damping_override=1`, and
  `van_der_waals_damping_override=9384`.
- The output still passes through `SelectedLineCatalog` and the common
  selected-line opacity kernel.

#### Water

- `read_water_line_catalog` yields three columns:
  wavelength code, signed lower-energy code, and signed strength code.
- The two signs select one of four source isotopologues.
- The exact offsets are `[-1, -3398, -2690, -5000]` in the packed
  log-strength convention.
- The selector requires atmosphere population slot 940 (zero-based row 939
  in the keep array), constructs packed species codes
  `-(9399 + isotope)`, and uses a dedicated compiled keep mask.
- Its radiative damping field is reconstructed from the current frequency-bin
  rule; packed Stark and van der Waals outputs are `1` and `9384`.
- It then joins the common selected-line representation and deposit.

#### H\(_3^+\)

- `generate_selected_lines` can read an explicitly supplied standard packed
  file and applies a `-1272` log-strength offset to every row,
  `population_slot_override=895`, `stark_damping_override=1`, and
  `van_der_waals_damping_override=9384`.
- `AtmosphereInput.h3plus_lines_path` therefore represents a runtime-capable
  explicit-path option.
- `source_line_paths()` does not return an H\(_3^+\) file. The standard
  high-level atmosphere source set therefore does not enable it.

The exact packed constants above are source contracts. Explain their role and
check their effects; do not rename them into apparently universal molecular
physics.

The standard high-level atmosphere workflow calls `source_line_paths()` and
therefore supplies diatomic, TiO, and water arrays. A stale `runner.py`
docstring says raw molecular selectors are off, but executable source
selection and the central runner guard contradict that sentence. The textbook
must follow executable behavior: converted diatomic/TiO/water paths are
active; H\(_3^+\) is explicit-path opt-in.

### 5.7 Availability and wiring are separate facts

Only after the reader has built the compilers and selectors should the exact
status matrix appear:

| line family | atmosphere source/selector | synthesis compiler | standard runtime behavior |
| --- | --- | --- | --- |
| diatomic / text bands | standard `diatomic_lines_path`; family correction then common selected deposit | `compile_molecular_text` | compiled and deposited when default `molecular_lines=True` |
| TiO | standard `titanium_oxide_lines_path`; family correction then common selected deposit | `compile_tio_schwenke` | compiled and deposited when default `molecular_lines=True` |
| water / H\(_2\)O | standard `water_lines_path`; dedicated selector then common selected deposit | `compile_h2o_partridge` | atmosphere active; synthesis compiler-only and absent from standard `_compile_molecular` |
| H\(_3^+\) | explicit `h3plus_lines_path` only; no standard source resolver entry | no standard compiler dispatch | atmosphere explicit-path opt-in; synthesis absent |

Also state:

- synthesis atomic lines are independent of `molecular_lines`;
- `molecular_lines=True` is the standard synthesis default and
  `molecular_lines=False` is the opt-out;
- the standard synthesis molecular builder concatenates manifest text arrays
  followed by TiO arrays;
- no public synthesis flag injects H\(_2\)O into that standard builder;
- `SPECIES_MASS_AMU[504] = 3.0` does not make H\(_3^+\) a compiled or
  deposited synthesis feature.

The machine-readable status artifact should encode separate fields such as
`compiler_exists`, `default_source_available`, and
`standard_runtime_deposits`. It is evidence, not a new public runtime API.

### 5.8 Common compiled fields and deterministic concatenation

Use `build_catalog_from_arrays`, not a new textbook dataclass. Check:

1. every compiler result has the same eight keys and one common length;
2. text results preserve manifest band order and source row order;
3. the standard synthesis concatenation is `[text, TiO]` for every field;
4. the combined `log_grid_ratio` and `grid_origin_index` match the exact
   `resolution` and requested start;
5. H\(_2\)O records remain absent from the standard combined result even
   after the compiler-only test has run.

The standard compiler re-indexes line centers later onto the exact
context-bearing synthesis wavelength array. Chapter 7 already explained why
the requested interior is not regenerated. Chapter 8 should show the
molecular re-index check once and not rederive the context-grid construction.

### 5.9 Cache is acceleration, not authority

Three cache layers must be distinguished:

1. **Per-source compiler caches.** Their keys include schema, window,
   `resolution`, wavelength policy, family options, and source-file identity.
2. **Combined standard molecular compiled cache.** Its identity includes the
   exact window, `resolution`, `use_energy_level_wavelengths=True`, compiler
   identity, and the text/TiO source identities. Its `__meta__` contains
   `cache_identity`, `start_wavelength_nm`, `end_wavelength_nm`, `resolution`,
   `log_grid_ratio`, `grid_origin_index`, and `n_lines`.
3. **Derived `MolecularLineCatalog` index cache.** Its metadata contains
   `schema`, `logic_version`, `log_grid_ratio`, and `grid_origin_index`;
   derived indices are rebuilt from the seven physical fields.

The `cache=True` on the Numba decorator is a fourth, lower-level cache of
compiled machine code, not a cache of molecular catalog values. A warm Numba
signature can remove compilation overhead even when the scientific
per-window array cache is empty. The timing cell must name which cache is cold
and which is warm.

A cache hit and a fresh build must produce value-identical catalog fields.
Corrupt or identity-mismatched caches are rebuilt. Deleting a cache changes
time, never physics.

The standard combined identity names only text-band and TiO sources. This is
additional executable evidence that H\(_2\)O is not in the standard synthesis
catalog.

Use “streaming” precisely. The text compiler advances one manifest band at a
time, and the canonical packed TiO/H\(_2\)O loaders memory-map their `.npy`
records. Some vectorized packed-field decoding still materializes full
derived columns; the pinned compiler is not a general fixed-size streaming
engine. The explicitly bounded `CHUNK_LINES`, `PAIR_CHUNK`, and offset blocks
belong to runtime opacity deposition in Movement 8B. Memory mapping, sequential
band processing, and runtime chunking solve related but different memory
problems.

The normal reader should demonstrate fresh and warm behavior in a disposable
local cache directory. Never read or write `~/payne-zero` at runtime. Chapter
10 will teach the broader in-process `WindowInvariants` cache composition and
multi-star residence; Chapter 8 owns only what makes the molecular compiled
product scientifically identifiable.

## 6. Movement 8B — Turn the compiled catalog into molecular opacity

### 6.1 Build `MolecularLineCatalog` and derived indices

From the seven physical arrays plus grid metadata, derive:

\[
\begin{aligned}
\texttt{center_index} &=
  \texttt{center_index_1based}-1,\\
\texttt{species_population_column} &=
  \texttt{species_code}//6-1,\\
\lambda_l &=
  \exp\!\left[
    \left(\texttt{center_index_1based}_l-1
    +\texttt{grid_origin_index}\right)
    \texttt{log_grid_ratio}
  \right].
\end{aligned}
\]

Check these expressions against `_precompute_indices` on a tiny catalog.
`unique_species_codes` gives the compact species inventory. Each line then
stores an integer `line_species_index` into that inventory after invariants
are built.

Explain why this is structure-of-arrays data: millions of line strengths,
centers, and species indices are contiguous fields, not millions of Python
objects.

### 6.2 Separate host decisions from device algebra

`precompute_invariants` performs one-time host work, then uploads reusable
tensors:

- reconstruct and re-index line centers on host;
- compute `local_resolving_power` in host float64, because adjacent geometric
  samples are too close for an early float32 difference;
- map unique species codes to exact population columns and masses;
- map every line to a compact species index;
- transfer indices, physical arrays, the wavelength grid, and Harris tables to
  the selected device.

The exact dtype contract must be visible:

- line strength and normalized damping enter in the package's established
  float32/default tensor precision;
- excitation, wavelengths, species masses, and profile evaluation use the
  backend work/high precision established in Chapter 2;
- indices use integer tensors;
- the shared opacity accumulator is float32 on CUDA, MPS, and CPU;
- CPU source parsing, Numba compilation, manifest/cache I/O, and memory mapping
  do not move to the GPU;
- no `torch.compile` path exists or is invented.

Show a small printed `field -> shape -> dtype -> device` table before the first
device kernel runs.

### 6.3 Population and molecular Doppler width

For each unique molecule:

\[
v_{\rm D}^2
=\frac{2kT}{m_{\rm mol}}+\xi^2,
\qquad
\frac{\Delta\nu_{\rm D}}{\nu}
=\frac{v_{\rm D}}{c}.
\]

Here `species_mass_amu` supplies \(m_{\rm mol}\) in atomic mass units,
converted with `ATOMIC_MASS_GRAM`; \(\xi\) is `microturbulence`.

The exact runtime ratio is

\[
R_{d,s}=
\frac{(n_s/U_s)_{d}}
     {\rho_d\,(\Delta\nu_{\rm D}/\nu)_{d,s}},
\]

returned with the Doppler fraction by
`species_population_doppler_ratio`.

Immediate checks:

- at zero microturbulence and fixed temperature, a heavier molecule has a
  smaller fractional width;
- at very large microturbulence, molecular-mass differences become small;
- doubling the stage-5 population doubles \(R_{d,s}\);
- doubling mass density halves \(R_{d,s}\);
- nonpositive population, density, or Doppler width produces zero rather than
  an invalid division;
- an unlisted species uses the exact conservative 20 amu default, while the
  status table makes clear that a mass entry alone is not source wiring.

This is a short reuse of Chapter 6 broadening logic, not a second Doppler
derivation.

### 6.4 The live continuum-relative keep test

For a line \(l\) and depth \(d\), gather its species ratio and compute:

\[
\begin{aligned}
A^{(0)}_{d,l}
 &= S_l R_{d,s(l)},\\
b_{d,l}
 &= \exp[-\tilde E_l\,(\texttt{hc_over_kt})_d],\\
A_{d,l}
 &= A^{(0)}_{d,l}b_{d,l},\\
a_{d,l}
 &= \frac{
   \gamma_{{\rm rad},l}
   +\gamma_{{\rm Stark},l}n_{e,d}
   +\gamma_{{\rm vdW},l}n_{{\rm pert},d}}
   {(\Delta\nu_{\rm D}/\nu)_{d,s(l)}}.
\end{aligned}
\]

The local opacity floor is the Chapter 7/production
`LINE_CENTER_CUTOFF_RATIO` times `continuum_opacity` at the clamped center.
The exact molecular keep mask requires:

- positive population ratio;
- positive Doppler fraction;
- positive line wavelength;
- `pre_excitation_strength >= opacity_floor`;
- positive `line_amplitude`;
- `line_amplitude >= opacity_floor`;
- nonnegative raw damping ratio.

Test each clause with one isolated boundary record. Do not call all of them
“selection”; this is the live depth-line keep after source compilation.

### 6.5 Three bounds for three temporary shapes

The reader should see why one chunk size cannot control every intermediate:

1. `CHUNK_LINES = 500_000` bounds dense `(depth, line)` cutoff arrays.
   `molecular_chunk_lines()` may read
   `PAYNE_ZERO_SYNTHESIS_MOLECULAR_CHUNK_LINES`, coerced to a positive integer.
2. Surviving depth-line pairs are flattened. `PAIR_CHUNK = 200_000` bounds
   pair-wise center and wing state.
3. Near and far wings walk integer offsets in blocks of 256, bounding
   `(pair, offset)` arrays.

The production algorithm loops over these bounded blocks in Python, while
each block's large algebra and sparse writes execute as Torch operations on
the resident device. Small scalar reads used to terminate bounded loops must
be named honestly; “device resident” does not mean that Python never controls
a loop.

Use a shape-and-memory table for one compact example. Do not turn this section
into Chapter 10's end-to-end hardware profile.

### 6.6 Centers, near wings, and far wings

Reuse Chapter 6's profile meaning and Chapter 7's sparse-deposit concept.
Chapter 8 adds only the molecular runtime order:

1. deposit the center at `center_index`;
2. walk the near wing out to ten Doppler widths, subject to
   `MAX_WING_PROFILE_STEPS`;
3. for small damping, use the established Harris table expression; for broad
   damping, use the established tabulated Voigt route;
4. include the first threshold-crossing sample, then mark that pair stopped;
5. if the profile survives to ten Doppler widths, seed the inverse-square far
   wing;
6. continue until the continuum floor, maximum step, or irreversible grid-edge
   break;
7. use `_scatter_add_flat` so overlapping pairs add rather than overwrite.

A tiny dense oracle may allocate `(D,L,W)` only for a handful of lines. It must
match the sparse center-plus-wing result within a predeclared float32
tolerance. Two lines aimed at one pixel must produce the sum of both
contributions.

Do not rederive Harris functions, far-wing asymptotics, or generic
scatter-add. Refer back once, then interpret what is molecular-specific:
population columns, molecular masses, and source compilation.

### 6.7 Apply stimulated emission once and form the component slab

`accumulate_molecular(..., apply_stim=True)` first completes all chunked
deposits in the shared float32 accumulator. It then multiplies by

\[
1-\exp\!\left(-\frac{h\nu}{kT}\right)
\]

and casts the result back to `ACCUMULATION_DTYPE`.

Check:

- `apply_stim=False` returns the gross slab;
- the ratio of net to gross equals the Chapter 5 factor wherever gross opacity
  is nonzero;
- applying the factor outside a standard `apply_stim=True` call would be a
  detectable double application;
- the final slab is finite and nonnegative.

Compute text-only, TiO-only, and standard text-plus-TiO synthesis slabs on the
same compact cool state. Their separately accumulated sum must match the
standard combined catalog result within the declared float32 accumulation
tolerance and standard concatenation order.

Run the H\(_2\)O compiler-only checkpoint separately. It may produce valid
compiled rows, but it must contribute exactly zero to the standard synthesis
slab because no standard runtime route concatenates it.

### 6.8 Close both lanes without conflating them

For the atmosphere fixture, run each family-specific reader/correction/
selector on compact records and pass the selected words into Chapter 7's
common decoding and deposition route. Verify:

- diatomic, TiO, and water each produce the exact selected-record and opacity
  contribution expected from the pinned source;
- standard high-level path resolution includes those three families;
- an explicit compact H\(_3^+\) path activates its selector;
- the standard resolver has no H\(_3^+\) entry;
- concatenating admitted molecular and atomic selected groups preserves the
  exact family order used by `generate_selected_lines`.

Do not implement the full atmosphere iteration here. Chapter 11 will compose
the standard source set, generate or load the common selected catalog on the
first opacity pass, and reuse the resulting objects later.

The final chapter checkpoint is therefore two lane-specific facts:

```text
atmosphere: active converted diatomic + TiO + water
            (+ H3+ only when an explicit path is supplied)

synthesis:  default-on manifest text + TiO device opacity
            (H2O compiler verified but not wired; H3+ absent)
```

Both are molecular line-opacity implementations. They are not identical
catalogs, source policies, or intermediate arrays.

## 7. Visible code-cell ledger

Target 17 substantial visible cells. Each cell should normally contain 10–30
lines, with 60 as a soft ceiling and 80 as a hard ceiling. Long exact kernels
belong in a canonical local runtime module; the notebook must still expose the
decisive equation, branch, or mask in a readable cell before calling the
optimized helper.

| cell | reader action | immediate visible result |
| ---: | --- | --- |
| 1 | verify the local manifest, source hashes, compact source slices, and reader-built cool state | provenance/shape table; no external checkout used |
| 2 | map H2, CO, TiO, and H2O species codes to the Chapter 4 stage-5 population lane | exact columns and one-population/many-lines stick plot |
| 3 | parse one fixed-width text record and compare stored versus energy-derived wavelength | every field named; fallback and predicted-line signs interpreted |
| 4 | compile a tiny manifest-ordered text subset with the readable scalar loop | accepted/rejected record trace and exact eight-array dtypes |
| 5 | run the serial cached-`njit` text kernel | record equality; scalar, cold, and warm timings kept separate |
| 6 | compile the compact 32-band manifest subset in declared order | per-band counts, source-row order, and combined fingerprint |
| 7 | decode a packed TiO slice | fine-grid center, isotope fraction, strength, and vacuum/air comparison |
| 8 | decode all four H2O sign cases and compile a packed slice | isotopologue table and compiler parity, no runtime claim |
| 9 | apply atmosphere diatomic and TiO family corrections before the common selector | exact offset/override effects and selected counts |
| 10 | run the atmosphere water selector and explicit-path H3+ probe | water active; H3+ explicit-path only |
| 11 | build the compiler/source/runtime feature-status table from audited call traces | standard synthesis excludes H2O and H3+; `molecular_lines=False` is opt-out |
| 12 | concatenate standard text then TiO arrays and build `MolecularLineCatalog` | exact fields, derived indices, center re-index check |
| 13 | test fresh/cached/corrupt-cache paths in a disposable directory | value identity and cache-key invalidation |
| 14 | precompute `MolecularLineInvariants` | field/shape/dtype/device table |
| 15 | calculate species population/Doppler ratios | mass, temperature, microturbulence, density, and population limits |
| 16 | compare the tiny dense oracle with chunked center/near/far sparse deposits | overlapping-pixel addition and chunk-size tolerance |
| 17 | form text-only, TiO-only, standard synthesis, and atmosphere family checkpoints | final two-lane status/parity table; no transfer or flux |

The exact H\(_2\)O compiler check belongs in cell 8 and its non-wiring proof in
cell 11/17. Do not hide either fact in a footnote.

If a cell merely calls `run_chapter08()` and prints a final answer, the
pedagogical contract has failed. A high-level helper is allowed only after the
reader has already seen the objects and decisions it composes.

## 8. Quantitative figures and original schematics

### 8.1 Professional one-claim quantitative figures

Use the book's shared paper-inspired typography, explicit palette, inward
ticks, white background, and unit-bearing axes. Every figure is deterministic,
one panel, and interpreted immediately.

1. **One population, many lines.** A stick spectrum of a compact CO band
   subset, with line height proportional to the declared line-strength
   quantity. Sole claim: one species population feeds many distinct
   transitions.
2. **Manifest order becomes catalog order.** Compiled line-center density or
   a restrained ordered-rug view for a few named text bands in one window.
   Sole claim: band concatenation follows manifest order while centers may
   overlap.
3. **Cool-state molecular opacity.** Standard text-band and TiO contributions
   across one compact wavelength interval at one named depth, with their sum.
   Sole claim: independent source families add to one molecular slab. Do not
   show flux.
4. **Population controls band strength.** Integrated molecular opacity for a
   controlled scaling of one stage-5 population, holding the atmosphere and
   source catalog fixed. Sole claim: the opacity scales with the supplied
   molecular population before saturation/transfer enters.

Compiler timing, cache identity, chunk errors, and the availability/wiring
matrix are better shown as compact tables. Do not manufacture plots for them.

Plot rules:

- wavelength in nm and opacity in cm\(^2\) g\(^{-1}\);
- name the depth or temperature used;
- use log axes only when necessary and label zero-handling explicitly;
- no positional Matplotlib colors or default style cycle;
- no multi-panel dashboard, gradients, decorative fills, or unexplained
  scientific-notation offset;
- direct annotation is preferred when only one or two curves need labels;
- inspect notebook-width, full-resolution, and mobile-reader renderings for
  clipping, label collisions, excess whitespace, and contrast.

### 8.2 Original conceptual schematics

Inspect the prompt architecture and visual grammar in
`/Users/ysting/payne-zero-website/scripts/generate_physics_images.py`, but do
not reuse a website image or import the website at reader runtime. Add new
chapter-owned specifications to the textbook's local schematic registry and
generate original compositions.

Follow the shared language:

- pure white background;
- hand-drawn scientist-notebook feel;
- slate blue, soft charcoal, warm grey, pale beige, and sparing deep navy;
- short audited labels, slightly varied line weight, and generous whitespace;
- landscape layout;
- no gradients, shadows, logos, watermarks, fake numbers, or decorative
  filler.

Required original compositions:

1. **One molecule becomes a band.** One molecular energy family fans into
   rotational/vibrational ladders and then many nearby absorption sticks.
   The scene is conceptual and must not imply measured spacings.
2. **Three encodings, one compiled record.** A text-band card, a TiO packed
   field strip, and an H\(_2\)O signed-field strip converge on the seven
   physical molecular line arrays plus one margin field. The labels teach
   semantic convergence, not byte archaeology.
3. **Host compiler to device chunks.** Manifest and packed sources are decoded
   on CPU; invariant arrays move once to a device; line chunks become
   surviving depth-line pairs; centers and wings add into a `(D,W)` slab.
4. **Two lanes, four statuses.** An atmosphere path shows standard diatomic,
   TiO, and water plus an explicit-path H\(_3^+\) branch. A synthesis path
   shows default text plus TiO, a compiler-only H\(_2\)O side branch, and no
   H\(_3^+\) source route. Keep the wording short enough to remain readable
   on mobile.

Each schematic needs:

- an owned prompt/specification;
- generation provenance and source-style reference;
- SHA-256 and dimensions;
- alt text that states the causal relationship rather than listing colors;
- a caption explicitly saying “conceptual” when sizes or paths are not
  measured;
- desktop, mobile, and full-resolution-reader inspection.

## 9. Self-contained source and data staging

The notebook must run from this repository alone. The pinned checkout and
paper remain read-only development oracles and are never reader dependencies.

### 9.1 Pinned source identities audited for this contract

| pinned source file | SHA-256 |
| --- | --- |
| `payne_zero_synthesis/source_catalog_molecular_compiler.py` | `b3e64b36f76228f9602490a927d7443dd701a36098573365da33e008436f633a` |
| `payne_zero_synthesis/molecular_lines.py` | `14c9d07e431fa73e6d6938e9db2d11c6688e52348234e0aac37cc76e8be3dc32` |
| `payne_zero_synthesis/pipeline.py` | `465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22` |
| `payne_zero_atmosphere/line_selection.py` | `b2c62fdf5e1fe43f33022184bfeff88985b13331354e3c745c7dab3a6b634fef` |
| `payne_zero_atmosphere/line_catalog.py` | `2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92` |
| `payne_zero_atmosphere/line_opacity.py` | `d0f9c43919be58a42547e12b7abc22161a7558bf17abbcd375ab04ccf57d7cc6` |
| `payne_zero_atmosphere/source_catalogs.py` | `a9ea21735c9d4964b785d76c89c9fc976a30ed75f8b6f9d4f7c6aaa4e77dae36` |
| `payne_zero_atmosphere/runner.py` | `05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7` |
| `payne_zero_atmosphere/config.py` | `51e19846fb81c832ae57334faf3da2c1e4fc2ef9edf6e08467ef7296e4640b45` |

At contract time the textbook does not yet stage the complete synthesis
`molecular_lines.py` or `source_catalog_molecular_compiler.py`. The
implementation pass must copy or progressively stage the exact required
definitions locally, bind their identities in the source manifest, and prove
that no import resolves to `/Users/ysting/payne-zero`.

### 9.2 Full-source identities and optional-data boundary

| pinned source data | records/size | SHA-256 |
| --- | --- | --- |
| `molecules/manifest.json` | 32 text bands; 3,884 bytes | `65131b3c07f093f062afed5547875b969da1e607dc46e3120332758d5e1c32c6` |
| `molecules/molecular_band_lines.npz` | 22,377,706 records; 256 arrays; about 1.2 GB | `39cb02310e109eca28190b41d82148173e8000ad6a968cc65ec14e6d2f6bd7df` |
| `molecules/titanium_oxide_lines.npy` | 37,744,499 records; about 576 MB | `7def3a02828fba3a917581343f0cdbc11d97fb335a69c69098d348a344d38ca8` |
| `molecules/water_lines.npy` | 65,912,356 records; about 503 MB | `3fd5f26886c8af68a1910606790358e4971b3d8831b3d0c1be493ca73f615442` |
| `lines/diatomic_lines.npy` | 12,488,322 `(N,4)` int32 records; about 191 MB | `3980781fa73d9b4e39d9fa5e53dbdf5ad326447753f6146d9040c33036a7ad67` |

These files belong to the checksum-verified optional full-data installation,
not the normal repository.

### 9.3 Minimal normal-book data

Stage:

1. the small molecular `manifest.json`, preserving all 32 band names, order,
   counts, and original provenance labels;
2. one compact combined text-band subset with at least one auditable source
   row per manifest band, at least one retained row for every production
   dispatch family represented by the subset, and additional rows in the
   chosen teaching window for CO/CN/CH/OH or another physically interpretable
   set;
3. compact contiguous TiO and H\(_2\)O packed-record slices, with original row
   indices, source hashes, record dtypes, and regeneration command;
4. a compact atmosphere diatomic/TiO/water/H\(_3^+\) selector fixture that
   covers every family correction and exact keep boundary;
5. the already staged synthesis Harris/line-profile tables;
6. a reader-built cool molecular state from the canonical Chapter 4 path, or
   a physically separate integration fixture if isolation is necessary.

If an integration fixture is used, it must contain only upstream computed
state and be labeled `fixture`; it cannot contain Chapter 8 outputs or golden
answers. A Chapter 4 golden may be opened only after the reader's Chapter 4
calculation and cannot be repurposed as Chapter 8 input.

Every subset records:

- source path and full source SHA-256;
- pinned commit;
- source row indices or deterministic selection rule;
- array names, shapes, dtypes, axes, and units;
- role (`static`, `subset`, `fixture`, or `golden`);
- generating command and whether regeneration needs optional full catalogs;
- output size and SHA-256.

No opaque archive may mix source records, computed atmosphere state, and
comparison outputs.

## 10. Main-text checks and acceptance gates

Checks appear immediately after the idea they establish. Strict arrays and
large comparisons may live in tests, but the rendered chapter must show the
scientific meaning and a concise status.

### 10.1 Molecular physics and population mapping

- one molecule maps to several line records in the teaching band;
- every used `species_code` maps to the exact stage-5 column;
- mapped columns agree with the Chapter 4 54-lane public contract;
- line excitation is applied once to the partition-normalized molecular
  population;
- population, density, molecular mass, temperature, and microturbulence limits
  follow Section 6.3.

### 10.2 Text compiler

- all 32 manifest band names and counts match the pinned manifest;
- compact compilation order equals manifest order and source row order;
- scalar and serial cached-`njit` outputs are record-by-record exact for every
  field;
- cold compilation and warm execution are reported separately;
- `prange` is absent from the synthesis compiler and no fabricated parallel
  result is shown;
- stored/energy-derived wavelength, predicted-line sign, unknown dispatch,
  isotope correction, window margin, and ground-label damping branches each
  have an isolated check;
- every output key has the exact dtype and common length.

### 10.3 Packed TiO and H\(_2\)O compilers

- packed record dtypes match the source contract exactly;
- fine log-wavelength reconstruction matches the scalar formula;
- TiO isotope decoding and all five fractions are checked;
- the vacuum/air flag changes centers by the exact source correction;
- all four H\(_2\)O sign combinations map to the correct isotope index and
  fraction;
- compact TiO and H\(_2\)O compiler outputs match the pinned source;
- the standard combined synthesis compiler contains text then TiO and no
  H\(_2\)O record.

### 10.4 Atmosphere molecular selectors

- canonical and legacy readers produce the same packed words on compact
  fixtures;
- diatomic isotope/source offsets and Stark override match exactly;
- TiO five-isotope offsets and all three exact overrides match;
- water sign decoding, population row, isotope offsets, and packed damping
  fields match;
- the family-specific keep masks equal scalar references at each strict or
  inclusive boundary inherited from Chapter 7;
- diatomic, TiO, and water are present in the standard source resolver;
- H\(_3^+\) contributes only with an explicit existing path and is absent from
  `source_line_paths()`;
- decoded products are valid `SelectedLineCatalog` objects and use the common
  deposit without a second family-specific profile.

### 10.5 Catalog, cache, device, and dtype

- derived catalog indices match independent scalar reconstruction;
- write/reload preserves the seven physical fields and rebuilds derived
  indices exactly;
- changing window, `resolution`, wavelength policy, band order, source
  identity, schema, or logic version invalidates the relevant cache;
- a corrupt cache rebuilds from declared source inputs;
- cache hit and fresh build produce identical scientific arrays;
- all host/device fields match the declared dtype and device table;
- local resolving power is computed from host float64 wavelength differences;
- no compiler, parser, manifest, or cache write occurs on the Torch device.

### 10.6 Sparse molecular opacity

- a tiny dense `(D,L,W)` oracle matches the sparse result within a
  predeclared dtype-specific tolerance;
- two lines at one target pixel add rather than overwrite;
- center, near-wing, and far-wing contributions are separately finite and
  nonnegative;
- the first below-floor near-wing sample and irreversible far-wing edge stop
  match the scalar reference;
- the standard `CHUNK_LINES`, `PAIR_CHUNK`, and offset block sizes bound the
  intended temporary shapes;
- two safe `chunk_lines` values give the measured float32 regrouping error;
  strict parity uses the standard chunk policy;
- `apply_stim=True` applies the factor exactly once and returns float32;
- text-only plus TiO-only agrees with standard combined molecular opacity in
  standard order;
- the final synthesis molecular slab matches the pinned same-state authority;
- the atmosphere family-separated checkpoints match their own pinned
  authorities, not the synthesis intermediates.

### 10.7 Feature boundary

- `molecular_lines=True` is confirmed as the standard synthesis default;
- `molecular_lines=False` removes standard molecular compilation/deposition
  but leaves atomic compilation/deposition eligible;
- `compile_h2o_partridge` succeeds on the teaching slice;
- an audited call trace proves standard `_compile_molecular` invokes only
  `compile_molecular_text` and `compile_tio_schwenke`;
- standard synthesis opacity remains H\(_2\)O-free;
- no H\(_3^+\) synthesis compiler or pipeline source route is claimed;
- a species-mass row alone cannot satisfy compiler or runtime wiring status.

Every tolerance must be declared beside the quantity, dtype, device, and
chunk policy before the comparison runs. Discrete codes, order, dtypes,
indices, status rows, and compiler outputs require exact equality. Floating
deposits use measured, branch-specific tolerances rather than the phrase
“machine precision.”

## 11. Redundancy, deferral, and language audit

Chapter 8 must not:

- rederive Chapter 4 mass action, molecular Newton solves, continuation, or
  public population packing;
- call the stage-5 molecular population an actual ion-stage population;
- divide by the molecular partition function a second time;
- rederive Chapter 6 ordinary line-strength, Doppler, damping, stimulation, or
  Voigt physics;
- reteach Chapter 7's generic window context, keep inequality, scatter-add, or
  race-condition lesson;
- describe a directory tree or production file in source order before the
  physical need;
- paste large source blocks into Markdown;
- call the text NPZ a runtime raw-text parser;
- imply that all molecular formats share a universal decoder;
- claim that the serial compiler uses `prange`;
- claim that synthesis uses `torch.compile`;
- call the converted atmosphere diatomic/TiO/water selectors unsupported;
- make the stale `runner.py` docstring override executable behavior;
- call H\(_3^+\) a standard atmosphere source or a synthesis feature;
- call H\(_2\)O a standard synthesis feature, public opt-in, or hidden
  contribution;
- describe `molecular_lines=False` as disabling atomic lines;
- call `resolution` instrumental resolution;
- treat a cache as a golden output or scientific input;
- teach Chapter 10's complete window-invariant cache/device lifecycle;
- teach Chapter 11's first-pass atmosphere selection and later object reuse;
- compute flux or normalize a spectrum before Chapter 9;
- add detached exercises.

Permitted backward references are minimal:

- one paragraph recalling Chapter 4's stage-5 molecular population;
- one sentence recalling Chapter 6's ordinary profile;
- one sentence recalling Chapter 7's common selected-record and sparse
  deposit machinery.

Permitted forward references are likewise minimal:

- Chapter 9 receives completed opacity but must still determine what escapes;
- Chapter 10 keeps the now-understood molecular invariants resident in the
  complete synthesis architecture;
- Chapter 11 composes the atmosphere source families and reuse lifecycle.

Do not use a later chapter as a substitute for explaining any object consumed
inside Chapter 8.

## 12. Chapter summary and causal handoff

End with a short summary whose claims can be read without the source tree:

1. Molecules produce bands because rotational, vibrational, and electronic
   structure gives one species many transitions.
2. The Chapter 4 stage-5 population is the thermochemical owner; each line
   contributes its own lower-state excitation, strength, and profile without
   repeating equilibrium.
3. Text bands, packed TiO, packed H\(_2\)O, and atmosphere converted families
   retain different source semantics before reaching common line fields.
4. The synthesis text compiler is a serial cached-Numba loop because emitted
   record order is part of the result; independent atmosphere keep masks may
   use `prange`.
5. Host compilation and exact discrete decisions build device invariants;
   bounded line, pair, and offset chunks then add centers and wings into one
   float32 molecular opacity slab.
6. Availability is not wiring: atmosphere water is standard, atmosphere
   H\(_3^+\) is explicit-path opt-in, synthesis text/TiO are default-on,
   synthesis H\(_2\)O is compiler-only, and synthesis H\(_3^+\) is absent.
7. The chapter ends with checked molecular mass-opacity contributions, not an
   emergent spectrum.

The final causal link to Chapter 9 should read in this spirit:

> We can now account for continuous opacity, atomic forests, special atomic
> profiles, and molecular bands at every depth and wavelength. These arrays
> tell us how strongly matter absorbs or redirects radiation, but not how much
> light escapes through the overlying layers. Chapter 9 follows radiation
> through that ordered depth structure and turns the completed opacity into
> total and continuum flux.

That is the required forward link. It closes the opacity part of the book and
makes radiative transfer the next physical necessity rather than the next
repository module.
