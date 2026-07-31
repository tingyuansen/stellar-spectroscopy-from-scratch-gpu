# Chapter 6 Fe I record candidate audit

Status: independent read-only candidate audit  
Audited candidate: zero-based raw row `873702`  
Verdict: **ACCEPT as the Chapter 6 teaching-record candidate; not yet frozen or
published**

## 1. Scope and decision

This audit independently checked the recommended row against the pinned
synthesis source archive, ran the pinned `_build_records` transformation, and
passed the resulting one-record mapping through the exact CPU synthesis
line-opacity route against continuum opacity recomputed from each of the four
six-depth Chapter 5 states.

The candidate passes the objective interpretation implied by the causal
outline: it is active in at least one supplied depth in every stellar regime,
and the chapter can report the active-depth count rather than silently choosing
a favorable layer. The exact counts are:

| regime | active depths / 6 | zero-based active mask |
| --- | ---: | --- |
| hot dwarf | 3 | `1, 1, 1, 0, 0, 0` |
| solar dwarf | 6 | `1, 1, 1, 1, 1, 1` |
| low-gravity giant | 6 | `1, 1, 1, 1, 1, 1` |
| cool molecule-rich | 6 | `1, 1, 1, 1, 1, 1` |

This is not an authorization to call the record, either lane's fixture, or
either golden “frozen.” The manifest-bound subset, synthesis golden,
provenance-bound atmosphere packing, and 80-depth atmosphere oracle remain
separate gates in the exact source contract.

## 2. Authorities and identities

The read-only source checkout resolved to the required commit:

`9c44001feae40b85146630499e6f8a5fed42e5af`

| authority | byte size | SHA-256 |
| --- | ---: | --- |
| `source_data_files/source_catalogs/lines/atomic_source_lines_parsed.npz` | 258,021,389 | `4eafa927c02a4f74401523149a44e35239f2aaecb4a64f2905a4cd5530c2dde7` |
| `payne_zero_synthesis/atomic_lines.py` | — | `0fa52833fb16487da1d5bfaaf5628a46751f888c1a57894a5037daa6d6667ab0` |
| `payne_zero_synthesis/line_opacity.py` | — | `639b95c3812f1a7d227b797fa89a4d6ef9725d5f0e1284f3d49cf86844278275` |
| `payne_zero_synthesis/pipeline.py` | — | `465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22` |
| synthesis `line_profile_tables.npz` | 57,308 | `87b47fc76bed10455218f43c4b6686525b961002e72d6a5ef01255a08deb27d4` |
| `data/fixtures/chapter05_continuum_states.npz` | 1,014,448 | `ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae` |
| staged synthesis `continuum_tables.npz` | — | `406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d` |
| staged synthesis `continuum_edge_grid.npz` | — | `11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e` |

The archive contains exactly 1,939,975 rows and exactly the 17 fields audited
below. Row `873702` is unique when all 17 fields are compared. It is also the
only row with the conjunction
`species_code == 26.0` and `stored_wavelength_nm == 499.0341`, and the only row
with the same species, two energy columns, and raw `log(gf)`.

No source, fixture, static table, or golden was written or changed during this
audit.

## 3. Every raw field

The dtype strings are NumPy's exact archive dtypes. Values are the exact scalar
values returned from row `873702`.

| raw field | dtype | exact value |
| --- | --- | --- |
| `stored_wavelength_nm` | `<f8` | `499.0341` |
| `raw_log_oscillator_strength` | `<f8` | `-0.86` |
| `species_code` | `<f8` | `26.0` |
| `first_energy_column_cm` | `<f8` | `53545.833` |
| `second_energy_column_cm` | `<f8` | `33507.123` |
| `radiative_damping_log` | `<f8` | `8.47` |
| `stark_damping_log` | `<f8` | `-4.45` |
| `van_der_waals_damping_log` | `<f8` | `-7.51` |
| `lower_principal_quantum_number` | `<i8` | `0` |
| `upper_principal_quantum_number` | `<i8` | `0` |
| `primary_isotope_number` | `<i8` | `0` |
| `primary_isotope_log_correction` | `<f8` | `0.0` |
| `secondary_isotope_log_correction` | `<f8` | `0.0` |
| `energy_shift_field` | `|S10` | `b'          '` (ten bytes `0x20`) |
| `isotope_shift_units` | `<f8` | `0.0` |
| `line_size` | `<i8` | `0` |
| `line_category_tag` | `|S3` | `b''` |

The two principal-quantum-number zeros are exact values in the parsed archive;
they should not be narrated as physical principal quantum number \(n=0\).
The parsed numerical archive cannot distinguish an originally blank numeric
fixed-width field from a textual zero. Neither value is consumed by the
ordinary line kernel in Chapter 6.

The two byte-string fields retain their distinct meanings:

- `energy_shift_field` is ten literal spaces. The exact two-subfield parser
  returns lower and upper energy shifts of `0.0 cm^-1`.
- `line_category_tag` is the stripped empty `S3` value. It is not `AUT`, `COR`,
  or `PRD`.

## 4. Zero corrections and exact derived record

The production transformation applies no hidden adjustment:

| transformation term | result |
| --- | ---: |
| primary isotope log correction | `0.0` |
| secondary isotope log correction | `0.0` |
| parsed first energy shift | `0.0 cm^-1` |
| parsed second energy shift | `0.0 cm^-1` |
| isotope wavelength shift | `0.0 nm` |

The absolute energy columns give

\[
E_{\rm lower}=33507.123\ {\rm cm}^{-1},\qquad
E_{\rm upper}=53545.833\ {\rm cm}^{-1},
\]

\[
\Delta E=20038.71\ {\rm cm}^{-1},\qquad
\lambda=\frac{10^7}{\Delta E}
=499.03411946178176\ {\rm nm}.
\]

The stored-minus-derived difference is
`-1.946178173284352e-05 nm`. Because the shifted energy difference is
positive, `_build_records` uses the energy-derived wavelength; it does not
fall back to `stored_wavelength_nm`.

The exact derived ordinary record is:

| derived field | dtype | exact value |
| --- | --- | ---: |
| `wavelength_nm` | `<f8` | `499.03411946178176` |
| `index_wavelength_nm` | `<f8` | `499.03411946178176` |
| `log_oscillator_strength` | `<f8` | `-0.86` |
| `oscillator_strength` | `<f8` | `0.1380384264602885` |
| `lower_excitation_cm` | `<f8` | `33507.123` |
| `atomic_number` | `<i8` | `26` |
| `ion_stage` | `<i8` | `1` |
| `line_type` | `<i8` | `0` |

`oscillator_strength` is \(gf\), not bare \(f\). The line kernel multiplies
it by `partition_normalized_populations`; inserting another lower-level
statistical weight would double count \(g_l\).

The nonzero raw damping logs give explicit coefficients:

| coefficient | exact raw value |
| --- | ---: |
| radiative \(\gamma\) | `2.9512092266663903e8 s^-1` |
| Stark coefficient | `3.5481338923357534e-5 cm^3 s^-1` |
| van der Waals coefficient | `3.090295432513592e-8 cm^3 s^-1` |

All three `need_*_default` predicates are false. At
`frequency_hz = 600745412604918.0`, division by the literal
`12.5664 * frequency_hz` produces:

| production field | exact host `float64` value | invariant upload |
| --- | ---: | ---: |
| `radiative_damping` | `3.909296919359072e-08` | `3.909297063842132e-08` `torch.float32` |
| `stark_damping` | `4.700008650504819e-21 cm^3` | `4.700008736577192e-21` `torch.float32` |
| `van_der_waals_damping` | `4.093536407068315e-24 cm^3` | `4.093536498989339e-24` `torch.float32` |

The parity-pinned classical coefficient gives

`classical_line_strength = 3.4403587659200408e-18 cm^2`

on the host and `3.4403587321228807e-18` after its required
`torch.float32` invariant upload.

These transformations establish ordinary Fe I semantics without relying on
the element name alone: `species_code == 26.0` maps to atomic number 26 and
neutral stage 1; the empty category is a standard line; Fe I is not rerouted
through the H, D, He, autoionizing, COR, or PRD branches; and all three damping
values are explicit.

## 5. Visibility audit setup

The visibility calculation used a declared, round-number wavelength window
chosen without inspecting line strength:

| quantity | value |
| --- | --- |
| requested window | `495.0` to `505.0 nm` |
| intrinsic sampling | exact `Grid.resolution = 300000` |
| exact `Grid.build()` result | `(6000,) float64` |
| first / last sample | `495.0009387906341 / 504.9989209057178 nm` |
| mapped center and wing-anchor index | `2434 / 2434` |
| mapped center sample | `499.0333758196059 nm` |
| distance from center to nearest boundary | more than 2,400 samples |

The maximum measured wing reach was only 163 samples, so no reported reach or
nonzero count is clipped by this audit window.

For each regime, the audit:

1. loaded the input-only six-depth synthesis state;
2. recomputed `continuum_absorption` and `continuum_scattering` through the
   Chapter 5 standard edge-triplet route on the audit grid;
3. formed `continuum_opacity = continuum_absorption + continuum_scattering`;
4. rebuilt the exact synthesis collision proxy from H I, He I, H2, and
   temperature;
5. called pinned `precompute_invariants` with one ordinary record, empty
   helium metadata, and all three exact Harris tables;
6. called pinned `accumulate_atomic(..., do_metal=True, do_helium=False)` with
   both `apply_stim=False` and `apply_stim=True`;
7. compared the `"batched"` and `"loop"` wing modes.

No Chapter 5 or Chapter 6 golden was opened. The cutoff continuum entered the
line kernel through its required `float32` fence.

The exact activity predicate was

```text
population > 0
fractional width > 0
mass density > 0
pre-excitation strength >= 1e-3 * continuum opacity
post-FASTEX line amplitude >= 1e-3 * continuum opacity
line amplitude > 0
```

Thus “visible” below means the production center gate is active. It is not a
hand-selected plotting threshold.

## 6. Four-regime result

This summary uses gross, pre-stimulated line opacity because that is the
quantity compared with the continuum by the center gate. The net route was
also executed and is reported in the depth table.

| regime | active / 6 | post-FASTEX strength / cutoff, active range | Doppler width, active range (nm) | damping ratio, active range | maximum gross line / continuum | maximum gross opacity (cm2 g-1) | reach range (samples) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hot dwarf | 3 | `4.50815–156.265` | `0.00420546–0.00467036` | `0.0134674–1.24755` | `0.153891` | `0.257539` | `5–6` |
| solar dwarf | 6 | `725.872–81754.3` | `0.00379272–0.00420546` | `0.00522449–8.53045` | `81.2725` | `4.73227` | `24–114` |
| low-gravity giant | 6 | `3211.38–28031.0` | `0.00373792–0.00405560` | `0.00514781–0.0919687` | `27.8682` | `2.19996` | `16–32` |
| cool molecule-rich | 6 | `866.209–21938.0` | `0.00365983–0.00400441` | `0.00542525–52.1971` | `21.7741` | `1.78584` | `15–163` |

The exact CPU `"batched"` and `"loop"` gross slabs are bitwise equal in all
four regimes: maximum absolute difference `0.0`. Every slab has shape
`(6, 6000)`, dtype `float32`, finite nonnegative values, one ordinary
invariant, zero autoionizing invariants, and no helium line. The population
indices are exactly ion-stage index `0` and element index `25`.

## 7. Depth-resolved cutoff evidence

Depth index increases from outermost to innermost. `post/cut` is the
post-FASTEX line amplitude divided by the exact float32-fenced center cutoff.
`center/cont` is the actually deposited gross center opacity divided by the
recomputed continuum. `reach` is the exact batched wing reach; the number of
nonzero pixels is `2 * reach + 1`.

| regime | depth | T (K) | post/cut | center/cont | damping ratio | reach | active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hot dwarf | 0 | 8000 | `156.265` | `0.153891` | `0.0134674` | 6 | yes |
| hot dwarf | 1 | 10000 | `27.8414` | `0.0235132` | `0.137820` | 6 | yes |
| hot dwarf | 2 | 13000 | `4.50815` | `0.00166621` | `1.24755` | 5 | yes |
| hot dwarf | 3 | 18000 | `0.284016` | `0` | `8.70688` | 0 | no |
| hot dwarf | 4 | 25000 | `0.0253652` | `0` | `58.2483` | 0 | no |
| hot dwarf | 5 | 32000 | `0.00904887` | `0` | `416.359` | 0 | no |
| solar dwarf | 0 | 4000 | `81754.3` | `81.2725` | `0.00522449` | 36 | yes |
| solar dwarf | 1 | 4500 | `44016.9` | `43.7285` | `0.00580802` | 28 | yes |
| solar dwarf | 2 | 5200 | `15331.6` | `15.1284` | `0.0117497` | 24 | yes |
| solar dwarf | 3 | 5800 | `6826.13` | `6.29107` | `0.0694896` | 40 | yes |
| solar dwarf | 4 | 6500 | `2933.37` | `1.62124` | `0.633171` | 80 | yes |
| solar dwarf | 5 | 8000 | `725.872` | `0.0476850` | `8.53045` | 114 | yes |
| low-gravity giant | 0 | 3500 | `16298.5` | `16.2025` | `0.00522005` | 16 | yes |
| low-gravity giant | 1 | 4000 | `21816.8` | `21.6900` | `0.00515195` | 19 | yes |
| low-gravity giant | 2 | 4500 | `28031.0` | `27.8682` | `0.00514781` | 22 | yes |
| low-gravity giant | 3 | 5000 | `19409.3` | `19.2837` | `0.00573760` | 19 | yes |
| low-gravity giant | 4 | 5500 | `10171.3` | `10.0329` | `0.0120614` | 20 | yes |
| low-gravity giant | 5 | 6500 | `3211.38` | `2.87823` | `0.0919687` | 32 | yes |
| cool molecule-rich | 0 | 2800 | `13072.0` | `12.9920` | `0.00542525` | 15 | yes |
| cool molecule-rich | 1 | 3200 | `21938.0` | `21.7741` | `0.00662048` | 21 | yes |
| cool molecule-rich | 2 | 3600 | `12136.7` | `11.7984` | `0.0247055` | 30 | yes |
| cool molecule-rich | 3 | 4200 | `6256.51` | `4.69345` | `0.278851` | 73 | yes |
| cool molecule-rich | 4 | 5000 | `2542.47` | `0.365736` | `3.80038` | 163 | yes |
| cool molecule-rich | 5 | 6000 | `866.209` | `0.00936104` | `52.1971` | 73 | yes |

The hot-dwarf failures are not numerical underflow:

- depth 3 passes the pre-excitation gate by a factor `4.13412`, then fails
  after its FASTEX excitation weight reduces the post-excitation ratio to
  `0.284016`;
- depths 4 and 5 already fail the pre-excitation gate, with pre/cut ratios
  `0.174405` and `0.0408391`, and remain below cutoff after excitation.

This gives the chapter an honest regime-dependent cutoff example without
selecting a different line or layer.

The net-to-gross center factors in active layers range from `0.891151` to
`0.999966`, following the exact wavelength-dependent stimulated-emission
application. The net result was never used to decide visibility.

## 8. Grid-stability check

The same fixed 495–505 nm window was rerun at the public archive API's default
intrinsic sampling, `resolution = 20000`. Its grid has 400 samples, and its mapped center is
`499.04420746429804 nm`. Despite the coarser center sample, the exact active
masks remain:

```text
hot dwarf            1 1 1 0 0 0
solar dwarf          1 1 1 1 1 1
low-gravity giant    1 1 1 1 1 1
cool molecule-rich   1 1 1 1 1 1
```

The high-resolution visibility conclusion is therefore not a one-pixel
selection accident. Nonzero pixel counts change, as they should when the
intrinsic grid changes; the physical center-gate decision does not.

## 9. Ambiguities and remaining gates

### Closed

- Source row, archive identity, all 17 raw values, dtypes, and byte fields are
  unambiguous.
- The two isotope-log corrections, two energy shifts, and wavelength isotope
  shift are exactly zero.
- No default damping branch is active.
- Fe I, \(gf\), ion-stage, element-index, and ordinary `line_type == 0`
  semantics are exact.
- The same record is visible somewhere in all four supplied regimes under the
  exact production cutoff.
- CPU batched and loop wing routes are bitwise equal for this one-line case.
- The active mask is unchanged between \(R_{\rm grid}=20000\) and \(300000\).

### Open but non-rejecting

1. **“Visible under the six-depth states” needs an explicit acceptance
   sentence.** The causal outline asks for an active-depth count, which
   supports the criterion “at least one active depth in every regime.” Under
   that criterion this candidate passes. If the intended criterion were
   “active at all six depths in all four regimes,” it would fail because the
   three innermost hot-dwarf layers are rejected by the exact cutoff. The
   implementation acceptance record should freeze the intended criterion
   rather than leave it implicit.
2. **The readable one-row subset does not yet exist.** It must preserve the
   exact row index, archive hash, all 17 fields, dtypes, and both byte strings,
   and must receive its own deterministic hash and manifest entry.
3. **The six-depth synthesis golden is not yet frozen.** Its generating
   command, exact grid, source/table/fixture hashes, gross/net lifecycle,
   output dtype, axes, hashes, and tolerance policy remain to be recorded.
4. **Atmosphere representation is still unproved.** This audit establishes
   synthesis source identity only. The provenance-bound conversion to one
   `SelectedLineCatalog`, all seven quantized fields, reconstructed wavelength,
   80-depth input fixture, serial-kernel oracle, and pre/post-stimulated
   goldens remain mandatory. No atmosphere–synthesis slab identity is implied.
5. **Parsed zeros are not original fixed-width bytes.** The principal quantum
   numbers, isotope number, and line-size value are exact parsed integers, but
   their pre-parse textual spelling is not retained in this NPZ. This does not
   affect the Chapter 6 ordinary kernel; it should be recorded rather than
   overinterpreted.

## 10. Final disposition

**ACCEPT** row `873702` for deterministic extraction and the next synthesis
oracle gate. It is source-clean, physically ordinary, correction-free,
explicitly damped, unique in the authority archive, visible in every declared
stellar regime under the objective production cutoff, and numerically useful
because its hot-dwarf depth failures expose the cutoff rather than hiding it.

Keep the term **candidate** until the subset, synthesis golden, atmosphere
packing, 80-depth fixture, oracle results, and manifest records have all
passed their separate reviews.
