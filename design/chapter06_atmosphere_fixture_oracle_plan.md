# Chapter 6 atmosphere fixture and oracle plan

Status: design only; the already-published raw teaching subset is an accepted
input, and this pass created no atmosphere fixture, golden, or canonical
publication

Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

Scientific authorities:

- `design/chapter06_exact_source_contract.md`;
- `design/chapter06_causal_outline.md`;
- the pinned read-only source files and data identities listed below.

This plan owns only the Chapter 6 **atmosphere** one-line integration fixture,
its one-record packed conversion, and its lane-specific oracle lifecycle. The
synthesis one-line fixture and oracle are a separate implementation task. It
consumes, but did not create or modify, the manifest-bound raw teaching subset
at `data/subsets/chapter06_fe_i_source_row_873702.npz`. No command in this
design pass wrote to the Payne Zero tree or the paper tree.

## 1. Decision

Use zero-based raw synthesis-source row `873702` as the physical Fe I teaching
record, but convert it independently into one atmosphere
`SelectedLineCatalog` record. The conversion is provenance-bound and lossy:
it proves how the same declared Fe I physics is represented for this controlled
kernel test. It does **not** claim that the synthesis and atmosphere source
catalogs have a shared row identity or common genealogy.

Build the 80-depth atmosphere fixture from the packaged modern structured solar
example

```text
/Users/ysting/payne-zero/examples/data/sun_structured_atmosphere.npz
```

through the pinned fixed-electron-density atmosphere handoff:

```text
ModelAtmosphere adapter
  -> prepare_structured_handoff_population_state
  -> prepare_opacity_state with line flags disabled
  -> exact packed population/width projections
  -> exact 30,000-point atmosphere grid and (80,344) threshold
```

The structured example is an upstream physical-state source, not a line
golden. The fixture will contain only the projections that the selected-line
kernel reads. An oracle capture retains hashes of the full `(80,1006)` arrays
and proves that the projected reconstruction gives a bitwise-identical line
slab.

Store the atmosphere oracle compactly:

- one sparse encoding of the full pre-stimulated `(80,30000)` float32 slab;
- one float64 value vector for the exact downstream stimulated view on the
  same support;
- full dense shape, support, and dense-array hashes.

This avoids publishing two mostly-zero dense arrays while preserving exact
reconstruction.

## 2. What this fixture is and is not

The fixture is an **input-only integration fixture**. It supplies a physically
plausible 80-layer state so Chapter 6 can isolate the selected-line kernel. It
is not:

- a Chapter 6 construction of a converged atmosphere;
- proof that the packaged structured example is itself an atmosphere-solver
  debug product;
- a cross-catalog identity map;
- a replacement for the six-depth synthesis teaching states;
- a line-selection result from a full atmosphere catalog;
- a golden output.

The normal narrative should use the compact six-depth synthesis neighborhood
for factor-by-factor pedagogy. The 80-depth fixture appears only after the
physics has been built, at the exact atmosphere parity boundary.

## 3. Pinned source identities

### 3.1 Critical code

| pinned source | SHA-256 | ownership in this plan |
| --- | --- | --- |
| `payne_zero_synthesis/atomic_lines.py` | `0fa52833fb16487da1d5bfaaf5628a46751f888c1a57894a5037daa6d6667ab0` | raw-row interpretation only |
| `payne_zero_atmosphere/constants.py` | `ac1f1fbd345dc816eb3e70a8f97ebebc7a4c744fd2759b32ec19f8c88d987036` | exact/reference constants |
| `payne_zero_atmosphere/atmosphere_io.py` | `95c4d2cab230f6925e9404639ecb05b25af8c0c85755ac1ca70d760156a8683e` | `ModelAtmosphere`, abundance and \(hc/kT\) conventions |
| `payne_zero_atmosphere/runtime_state.py` | `fae240ec00f6f89d7c2a7ef721ce6e6539be234e523291fd6e8a096d731430e8` | runtime-state allocation |
| `payne_zero_atmosphere/population_layout.py` | `36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0` | Fe I packed slot |
| `payne_zero_atmosphere/equation_of_state.py` | `719b316327fd6fb76dfa5267a2c9022d1384c269a0fa20d5b2ec733671ae3fa2` | fixed-\(n_e\) atomic state |
| `payne_zero_atmosphere/molecular_data.py` | `705c3072d79c8019c948ce0fa2c82052f232816d453e10a7c8e5fc5a8f5ce249` | atmosphere molecular catalog |
| `payne_zero_atmosphere/molecular_equilibrium.py` | `4c9665148a57fadf4837f193c9f4247fbab44b788e7fb76ef3cae7ebdb3c3d86` | fixed-\(n_e\) molecular state |
| `payne_zero_atmosphere/doppler.py` | `e118a78bf5250ef5e1f77d652c9e78fbb7b92acf5c069f717faed7a3b3ea98f0` | packed widths and line support |
| `payne_zero_atmosphere/continuum_opacity.py` | `1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81` | atmosphere grid and threshold |
| `payne_zero_atmosphere/runner.py` | `05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7` | fixed-state/opacity composition and scalar stimulation check |
| `payne_zero_atmosphere/line_profile_math.py` | `9a5794140f00ff3c3fb6c2e3b28461bbc22b471f962d275055c066ad7f8acd15` | TABLOG, FASTEX, Harris basis |
| `payne_zero_atmosphere/line_catalog.py` | `2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92` | exact selected-record container |
| `payne_zero_atmosphere/line_selection.py` | `b2c62fdf5e1fe43f33022184bfeff88985b13331354e3c745c7dab3a6b634fef` | packed-format corroboration only |
| `payne_zero_atmosphere/line_opacity.py` | `d0f9c43919be58a42547e12b7abc22161a7558bf17abbcd375ab04ccf57d7cc6` | serial selected-line oracle |
| `payne_zero_atmosphere/transfer_kernels.py` | `50e759a085e6aefdb7819a3dbe3ef5e83405834f4b07e0a4de2f3c0e7354d3b9` | downstream stimulation boundary |

The capture worker must verify these identities before importing the modules
that perform the calculation.

### 3.2 Source data

| read-only source | bytes | SHA-256 | use |
| --- | ---: | --- | --- |
| `data/subsets/chapter06_fe_i_source_row_873702.npz` | `8665` | `bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04` | canonical manifest-bound raw row 873702 |
| `source_data_files/source_catalogs/lines/atomic_source_lines_parsed.npz` | `258021389` | `4eafa927c02a4f74401523149a44e35239f2aaecb4a64f2905a4cd5530c2dde7` | optional full-source regeneration witness |
| `source_data_files/source_catalogs/lines/observed_atomic_lines.npy` | `26812832` | `649b039474ca6cda6e9fe0dea3052e5d47c24320ba8be1f48cd2ec39b7bcf84f` | non-authoritative packing corroboration |
| `examples/data/sun_structured_atmosphere.npz` | `1633632` | `d686ea7107d60bf1707607e3d6377d283fb3eb7115c170ac2aeef54fbaa6abdb` | 80-depth source structure |
| `source_data_files/source_catalogs/lines/molecular_equilibrium_atmosphere.npz` | `19040` | `971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644` | molecular fixed-state handoff |
| `source_data_files/atmosphere_tables/isotope_tables.npz` | `169568` | `53c8d315fb53f1e051dc2752b028fc270d7c17a2c1042279c04ffcb750aef5c6` | masses and packed widths |
| `source_data_files/atmosphere_tables/iron_group_partition_tables.npz` | `45573` | `137629dea64eca46f77ea3656c18305ade912a468d7eb27029544c0106cc3296` | Fe-group state |
| `source_data_files/atmosphere_tables/ionization_potential_tables.npz` | `8292` | `82a2e82f2015da02c3d2bce77ca5337aa2b9c4e23d8d6219da07895896ca8a50` | atomic state |
| `source_data_files/atmosphere_tables/packed_level_metadata.npz` | `17816` | `de5f17b6a9eaec1d1b07e96fd02ff014279cd8eaa9f976fefde0e2a153961bc3` | packed state |
| `source_data_files/atmosphere_tables/special_partition_tables.npz` | `11364` | `7d737524aacda1cc2281e5b18ff49f240ca34665dbe6c96d4dd0f39db4aedd22` | atomic state |
| `source_data_files/atmosphere_tables/molecular_equilibrium_tables.npz` | `1935` | `1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe` | continuum H2 policy |
| `source_data_files/atmosphere_tables/continuum_opacity_tables.npz` | `57456` | `6fd4c556418870c28d3fcc9a050252af58ac4cc433cae979477355c8c7d593e3` | continuum threshold |
| `source_data_files/atmosphere_tables/karzas_latter_tables.npz` | `16826` | `23805dc17c47af45b8ae63b2e278e1fb6c584a01c87d1eb3c31306e4555e6d15` | continuum threshold |
| `source_data_files/atmosphere_tables/line_opacity_tables.npz` | `1359` | `89f486122cb8939b23dc5423145a46d88a77df8daf57a1def35055b7b8205f16` | atmosphere Harris basis |

The raw subset was produced by
`scripts/build_chapter06_fe_record_subset.py`, whose accepted SHA-256 is
`25bcf4662740155e8b08615b9522f3f4517e1a5ddc4627c68686620ccfff4d6c`.
Its schema version is one, and all 17 raw fields and all provenance members are
already registered in `data/MANIFEST.json`.

The final worker should also log the complete dynamic read set. An undeclared
additional source file is a capture failure, not silently added provenance.
The full raw archive and observed packed catalog are optional audit witnesses,
not required reader inputs. If either witness is supplied during capture, its
identity and result enter the full capture fingerprint; its absence may not
change the physical payload fingerprint.

`continuum_level_tables.npz` is intentionally absent: Chapter 5 established
that its loader is inactive in this continuum route. The worker must fail if
that archive is unexpectedly read rather than laundering an inactive table
into Chapter 6 provenance.

## 4. Provenance-bound conversion of raw row 873702

### 4.1 Raw authority

The canonical raw-subset row has:

```text
stored_wavelength_nm                 499.0341
raw_log_oscillator_strength          -0.86
species_code                         26.0
first_energy_column_cm               53545.833
second_energy_column_cm              33507.123
radiative_damping_log                8.47
stark_damping_log                    -4.45
van_der_waals_damping_log            -7.51
lower_principal_quantum_number       0
upper_principal_quantum_number       0
primary_isotope_number               0
primary_isotope_log_correction       0.0
secondary_isotope_log_correction     0.0
energy_shift_field                   b"          "
isotope_shift_units                  0.0
line_size                            0
line_category_tag                    b""
```

The conversion must first recheck all seventeen raw source fields and their
dtypes:

```text
float64: stored_wavelength_nm, raw_log_oscillator_strength, species_code,
         first_energy_column_cm, second_energy_column_cm,
         radiative_damping_log, stark_damping_log,
         van_der_waals_damping_log, primary_isotope_log_correction,
         secondary_isotope_log_correction, isotope_shift_units
int64:   lower_principal_quantum_number, upper_principal_quantum_number,
         primary_isotope_number, line_size
|S10:    energy_shift_field
|S3:     line_category_tag
```

In particular, the blank `energy_shift_field` remains ten literal space bytes
and the blank `line_category_tag` remains three zero bytes. Neither is
converted to an invented numeric zero. If the optional full archive is
available, deterministic re-extraction must reproduce the accepted subset
bytes exactly; conversion itself reads the subset rather than a hidden catalog
row.

### 4.2 Exact quantizers

Define

```python
ratio_log_step = np.log(1.0 + 1.0 / 2_000_000.0)

def packed_wavelength_code(wavelength_nm):
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError("wavelength must be finite and positive")
    code = int(np.floor(np.log(wavelength_nm) / ratio_log_step + 0.5))
    if code < np.iinfo(np.int32).min or code > np.iinfo(np.int32).max:
        raise ValueError("packed wavelength does not fit int32")
    return np.int32(code)

def tablog_code(positive_value):
    if not np.isfinite(positive_value) or positive_value <= 0.0:
        raise ValueError("TABLOG input must be finite and positive")
    code = int(np.floor(np.log10(positive_value) * 1000.0 + 16384.5))
    if code < 1 or code > np.iinfo(np.int16).max:
        raise ValueError("positive TABLOG index does not fit int16")
    return np.int16(code)
```

Both helpers must reject nonfinite or nonpositive inputs. The lookup has 32,768
entries, but this selected-record interface stores indices in signed `int16`
and the compiled kernel rejects nonpositive decoded values. Therefore this
converter fails closed unless the actual code is in `1..32767`; it does not
silently wrap conceptual lookup index 32,768 to `-32768`. The wavelength code
must fit `int32` before casting.

For this zero-shift row:

```text
lower_excitation_cm = min(abs(first_energy), abs(second_energy))
wavelength_nm = 1e7 / abs(first_energy - second_energy)
gf = 10**(raw_log_gf + primary_correction + secondary_correction)
gamma_rad = 10**radiative_damping_log
gamma_stark = 10**stark_damping_log
gamma_vdw = 10**van_der_waals_damping_log
```

The damping codes are formed from the three **raw** positive damping
quantities. They are not formed from the synthesis fields after division by
`12.5664 * frequency_hz`.

For Fe I:

```text
atomic_population_slot_start(26) = 350       # zero based
selected population slot          = 351       # one based
packed_species_slot               = 3510
```

The trailing zero is the controlled ordinary atomic encoding for this
one-record conversion. The kernel-relevant assertion is
`abs(packed_species_slot) // 10 == 351`.

### 4.3 Frozen conversion result

| `SelectedLineCatalog` field | value | dtype | canonical C-byte SHA-256 |
| --- | ---: | --- | --- |
| `packed_wavelength_index` | `12425352` | `(1,) int32` | `c423f4ac4a3825c6fad5336a1e15c0038ab17087f930916ad550ad45b4990dfc` |
| `packed_species_slot` | `3510` | `(1,) int16` | `c2126161f7488b7d198ea310da9a8694786a18a2317eea5e30379ee118d34743` |
| `lower_excitation_index` | `20909` | `(1,) int16` | `9b5bf0b74e2e212b57bcb2b9f2712eaab4c6169595f4e82767793f4534365648` |
| `log_strength_index` | `15524` | `(1,) int16` | `3fe821d54660a0c51b42d19a42571e208791b3c5ff6bc0cac16b6553a226515a` |
| `radiative_damping_index` | `24854` | `(1,) int16` | `5cdf3b75730a5b45c3f24da2c1030103143981191d2844aa7374e948b9abaeea` |
| `stark_damping_index` | `11934` | `(1,) int16` | `91d82e7b89ae43b29bb673bb416697d005e093d0011c9d7af501630c6a502141` |
| `van_der_waals_damping_index` | `8874` | `(1,) int16` | `c0346f09a0362e8e9d29c01a9fc7c292a7395b524a90319bb921608b9fdb1b60` |

The corresponding native four-word selected-line payload is

```text
[[12425352, 1370295734, 1628847268, 581578398]]
```

with C-byte SHA-256
`1769c9ad8d33e847a099bd6d50df85a2f478f98d554b2fc39db8121ba93158d2`.
Decoding those words with `detect_swapped_layout=False` must reproduce the
seven arrays exactly.

The wavelength code reconstructs
`499.03410878793585 nm`, which is
`-1.0673845906694623e-05 nm` from the unquantized energy-derived center. The
decoded float32 TABLOG values used by the kernel are:

```text
lower excitation       33496.54296875 cm^-1
gf                     0.13803842663764954
raw radiative gamma    295120928.0 s^-1
raw Stark coefficient  3.548133827280253e-05 cm^3 s^-1
raw vdW coefficient    3.090295308538771e-08 cm^3 s^-1
```

After the selected-line kernel's literal float32 order, the controlled record
has:

```text
classical strength     3.440358938918034e-18 cm^2
radiative damping      3.909297063842132e-08
Stark damping          4.700008332680409e-21 cm^3
vdW damping            4.093536498989339e-24 cm^3
```

These values are not expected to equal the readable synthesis record
bit-for-bit; the atmosphere payload has deliberately quantized wavelength,
excitation, strength, and damping.

### 4.4 Corroboration is not identity

A read-only diagnostic found the same seven decoded values at zero-based row
`780108` of `observed_atomic_lines.npy`. That row's four packed words equal the
payload above.

This is useful evidence that the quantizers and halfword order are correct.
It is **not** sufficient evidence that synthesis row 873702 and atmosphere row
780108 share an upstream catalog identity: the packed representation is lossy,
and no source-record genealogy has been established. Therefore:

- the fixture provenance begins at raw row 873702 plus conversion version;
- observed row 780108 is recorded only as
  `non_authoritative_packing_corroboration`;
- the observed catalog is not a reader runtime dependency;
- no atmosphere–synthesis slab equality is asserted.

## 5. Honest 80-depth fixture construction

### 5.1 Source structure and adapter

Read these arrays from the packaged structured example:

- `temperature (80,)`, K;
- `column_mass (80,)`, g cm\(^{-2}\);
- `gas_pressure (80,)`, dyn cm\(^{-2}\);
- `electron_density (80,)`, cm\(^{-3}\);
- `microturbulence (80,)`, cm s\(^{-1}\);
- `elemental_abundances (99,)`, linear number fractions.

Construct the exact atmosphere abundance boundary:

- H and He remain linear;
- Z=3…99 use `log10(elemental_abundances[Z-1])`;
- decoding with `linear_elemental_abundances` must recover all 99 source
  values exactly.

The source example does not carry all fields required by the
`ModelAtmosphere` constructor. Supply validator-only adapter fields:

```text
rosseland_opacity       ones(80), float64, positive placeholder
radiative_acceleration  zeros(80), float64
convective_flux         zeros(80), float64
convective_velocity     zeros(80), float64
```

These arrays are not published in the Chapter 6 fixture. A perturbation test
must show they do not alter any captured fixture member or line output in the
fixed-state opacity preparation used here.

Use the explicit controlled configuration:

```text
effective_temperature      5778.0 K
log_surface_gravity        4.44
pressure_iteration_enabled false
enable_molecules           true
enable_convection          false
temperature_iteration_index 1
```

Start from `DEFAULT_OPACITY_FLAGS`, set zero-based flags 14 and 16 to zero, and
leave the continuum flags unchanged. This computes the exact Chapter 5
continuum and threshold without selecting or accumulating any line. The one
controlled selected record is passed separately afterward.

The source filename says “sun,” but the file does not embed a complete
modern convergence/label record. The values above are therefore named
**fixture configuration**, not recovered source metadata, and the fixture is
not called a newly converged atmosphere.

### 5.2 Exact route

The worker must call, in order:

```python
population_state = prepare_structured_handoff_population_state(
    config,
    temperature_iteration_index=1,
)
opacity_state = prepare_opacity_state(
    config,
    population_state=population_state,
    temperature_iteration_index=1,
)
line_state = accumulate_selected_line_opacity(
    selected_lines=one_record,
    opacity_wavelength_grid_nm=opacity_state.opacity_wavelength_grid_nm,
    wavelength_bin_edges=opacity_state.wavelength_bin_edges,
    continuum_line_selection_threshold=(
        opacity_state.continuum_line_selection_threshold
    ),
    temperature=population_state.setup.atmosphere.temperature,
    hc_over_kt=population_state.setup.atmosphere.hc_over_kt,
    electron_density=population_state.runtime_state.electron_density,
    ion_stage_populations_by_packed_slot=actual_populations,
    partition_normalized_population_over_mass_density_and_fractional_doppler_width=(
        line_support
    ),
    fractional_doppler_widths=fractional_widths,
)
```

The fixed-state handoff is important. It recomputes the atmosphere packed
state from the structured columns without pretending that the public
`(80,6,139)` synthesis cube can be inverted into the atmosphere's packed
schedule.

### 5.3 Published fixture arrays

Proposed canonical path:

```text
data/fixtures/chapter06_atmosphere_one_line_inputs.npz
```

The raw teaching subset deliberately contains source fields and provenance
only. The seven packed arrays are computed atmosphere-kernel inputs, so they
live once in this fixture and bind the raw subset hash and conversion version.
They must not be added to, or mislabelled as part of, the raw subset.

| fixture member | shape | dtype | unit/convention | owner |
| --- | --- | --- | --- | --- |
| `packed_wavelength_index` | `(1,)` | `int32` | logarithmically packed wavelength | provenance-bound converter |
| `packed_species_slot` | `(1,)` | `int16` | ordinary Fe I packed slot `3510` | provenance-bound converter |
| `lower_excitation_index` | `(1,)` | `int16` | TABLOG lookup index | provenance-bound converter |
| `log_strength_index` | `(1,)` | `int16` | TABLOG lookup index | provenance-bound converter |
| `radiative_damping_index` | `(1,)` | `int16` | TABLOG lookup index | provenance-bound converter |
| `stark_damping_index` | `(1,)` | `int16` | TABLOG lookup index | provenance-bound converter |
| `van_der_waals_damping_index` | `(1,)` | `int16` | TABLOG lookup index | provenance-bound converter |
| `temperature` | `(80,)` | `float64` | K, outer to inner | structured source |
| `hc_over_kt` | `(80,)` | `float64` | cm, atmosphere rounded \(h,c,k\) convention | `ModelAtmosphere` property |
| `electron_density` | `(80,)` | `float64` | cm\(^{-3}\), fixed-state runtime value | atmosphere handoff |
| `actual_population_slot_indices` | `(3,)` | `int16` | zero-based `[0,2,840]` = H I, He I, H2 | projection descriptor |
| `actual_population_slot_values` | `(80,3)` | `float64` | cm\(^{-3}\) | atmosphere packed state |
| `line_population_slot_zero_based` | scalar | `int16` | `350`, Fe I | population layout |
| `partition_normalized_population_over_mass_density_and_fractional_doppler_width_at_line_slot` | `(80,)` | `float64` | g\(^{-1}\) | Doppler/strength state |
| `fractional_doppler_widths_at_line_slot` | `(80,)` | `float64` | \(\Delta v_D/c\) | Doppler state |
| `opacity_wavelength_grid_nm` | `(30000,)` | `float64` | nm, increasing direct atmosphere grid | Chapter 5 atmosphere grid |
| `wavelength_bin_edges` | `(344,)` | `int64` | packed wavelength codes; final sentinel `2**30` | Chapter 5 threshold grid |
| `continuum_line_selection_threshold` | `(80,344)` | `float32` | gross pre-stimulated line cutoff, cm\(^2\) g\(^{-1}\) | Chapter 5 atmosphere continuum |
| `effective_temperature` | scalar | `float64` | `5778.0` K, controlled grid configuration | fixture configuration |

These are the fixture's exact 19 array members. Do not copy raw-subset fields,
full packed arrays, validator-only adapter fields, line output, or a golden
into this NPZ.

The manifest entry and detached fixture acceptance record must bind:

- source commit and every source/data hash;
- source row, raw-subset schema, and conversion version;
- teaching-subset file hash and builder hash;
- fixture builder and worker hashes;
- the full source `(80,1006)` array fingerprints;
- units, axes, ownership, and reconstruction rules;
- Python, NumPy, and Numba versions;
- every fixed process control.

### 5.4 Reconstruction without false ownership

The fixture loader reconstructs the public-call arrays:

```python
selected_lines = SelectedLineCatalog(
    packed_wavelength_index=fixture["packed_wavelength_index"],
    packed_species_slot=fixture["packed_species_slot"],
    lower_excitation_index=fixture["lower_excitation_index"],
    log_strength_index=fixture["log_strength_index"],
    radiative_damping_index=fixture["radiative_damping_index"],
    stark_damping_index=fixture["stark_damping_index"],
    van_der_waals_damping_index=fixture["van_der_waals_damping_index"],
)

actual = np.zeros((80, 1006), dtype=np.float64)
actual[:, fixture["actual_population_slot_indices"]] = (
    fixture["actual_population_slot_values"]
)

line_support = np.zeros((80, 1006), dtype=np.float64)
line_support[:, 350] = fixture[
    "partition_normalized_population_over_mass_density_and_"
    "fractional_doppler_width_at_line_slot"
]

fractional_widths = np.zeros((80, 1006), dtype=np.float64)
fractional_widths[:, 350] = fixture[
    "fractional_doppler_widths_at_line_slot"
]
```

This is a kernel-input projection, not a general atmosphere state. The worker
must run the selected-line call once with all three full upstream arrays and
once with the reconstructed projections, then require:

```text
same selected_line_count
same output dtype and shape
np.array_equal(full_output, projected_output)
same dense C-byte SHA-256
```

The full width array contains structural infinities at massless/unused slots.
The public call sanitizes them to zero. The projected fixture retains only the
finite Fe I width actually read by this record.

## 6. Oracle products

### 6.1 Pre-stimulated product

The selected-line accumulator returns:

```text
line_mass_absorption_coefficient (80,30000)
NumPy float32
cm^2 g^-1
pre-stimulated
selected_line_count = 1
```

This is the canonical atmosphere line product.

### 6.2 Downstream stimulated view

For each depth \(d\) and atmosphere wavelength index \(i\), compute the exact
transfer-input view

\[
s_{d,i}=\max\left[
1-\exp\left(-\nu_i h/(kT_d)\right),10^{-300}
\right],
\]

using the pinned atmosphere reference constants:

```text
frequency_hz = 2.99792458e17 / opacity_wavelength_grid_nm
h_over_kt = 6.6256e-27 / (temperature * 1.38054e-16)
post_stimulated = pre_stimulated.astype(float64) * stimulated
```

The oracle worker should call the pinned
`payne_zero_atmosphere.runner._planck_source_and_stimulated_emission` helper
for an independent column check, then compare its stimulated columns to the
broadcast construction.
The post-stimulated view is float64 because the transfer calculation multiplies
the float32 line slab by its float64 stimulated array.

This view is comparison evidence for the downstream boundary, not a second
stored `LineOpacityState`.

### 6.3 Compact golden encoding

Proposed canonical path:

```text
data/golden/payne_zero/chapter06/
  chapter06_atmosphere_one_line_cpu.npz
```

Store:

| golden member | shape | dtype | unit/convention | owner |
| --- | --- | --- | --- | --- |
| `dense_shape` | `(2,)` | `int64` | axis lengths `[80,30000]` | sparse oracle schema |
| `nonzero_flat_index` | `(240,)` | `int64` | zero-based C-order flat index | sparse encoder |
| `pre_stimulated_nonzero_value` | `(240,)` | `float32` | cm\(^2\) g\(^{-1}\), gross line opacity | selected-line kernel |
| `post_stimulated_nonzero_value` | `(240,)` | `float64` | cm\(^2\) g\(^{-1}\), once stimulated | transfer-boundary view |
| `selected_line_count` | scalar | `int64` | dimensionless exact value one | selected-line kernel |
| `gate_depth_1based` | `(10,)` | `int16` | one-based `[8,16,...,80]` | pinned gate topology |
| `gate_active` | `(10,)` | `bool` | exact candidate value: all true | independent oracle diagnostic |
| `dense_pre_stimulated_sha256` | scalar | `|S64` | lowercase ASCII SHA-256 | reconstructed pre slab |
| `dense_post_stimulated_sha256` | scalar | `|S64` | lowercase ASCII SHA-256 | reconstructed post slab |

These are the golden's exact nine array members. The 240-value support is an
exact-schema proposal from the feasibility witness below. If either fresh
capture produces another support size, stop and review the physical cause; do
not silently generalize the schema back to symbolic \(N\).

Reconstruction fills zeros in the declared dtype and assigns the value vectors
at `nonzero_flat_index`. It must reject:

- unsorted or duplicate indices;
- indices outside `0..80*30000-1`;
- different pre/post vector lengths;
- nonfinite or negative values;
- a reconstructed hash mismatch.

The pre/post arrays must have identical support because the stimulated factor
is strictly positive.

## 7. Required fingerprints

Use three distinct identities.

### 7.1 Member SHA-256

For every array:

```python
sha256(np.ascontiguousarray(array).tobytes(order="C"))
```

The manifest separately binds the array name, `dtype.str`, and shape, so byte
hashes are never interpreted without their schema.

### 7.2 Schema digest

Iterate over sorted member names and hash:

```text
name UTF-8 bytes
dtype.str UTF-8 bytes
shape as contiguous int64 bytes
```

This follows the established Chapter 5 schema-digest policy.

### 7.3 Capture fingerprints

Keep the two lifecycle phases distinguishable:

- **Fixture payload fingerprint:** sorted converted record, full-array source
  hashes, published projections, grid, threshold, source-state identity, and
  projection-parity decisions. Exclude only `meta__*` and `identity__*`.
- **Oracle payload fingerprint:** canonical fixture file/schema/member
  identities, pre-stimulated slab, stimulated array/view, support, selected
  count, gate decisions, and seam evidence. Exclude only `meta__*` and
  `identity__*`.
- **Full capture fingerprint:** all members of the corresponding raw capture,
  including source paths, module/table identities, environment, versions,
  command, conversion identity, and optional-witness presence. Exclude only
  self-referential fingerprint fields.

All fingerprints hash name, dtype, shape, and contiguous bytes. They must be
mapping-order independent. The fixture and oracle fingerprints must not share
a generic name that could let one phase's authorization satisfy the other.

The fixture raw capture may contain the full `(80,1006)` arrays. Its publisher
must not copy those arrays into the final projected fixture. The oracle raw
capture starts from the canonical fixture and must not need the full arrays.

## 8. Fresh-process oracle lifecycle

### 8.1 Fixture capture worker

Implement a read-only fixture worker such as:

```text
scripts/chapter06_atmosphere_fixture_worker.py
```

The fixture worker:

1. verifies the pinned commit by reading `.git/HEAD`/object identity without
   invoking filters or writing in the source tree;
2. verifies every declared source/module/data hash;
3. requires a new empty `NUMBA_CACHE_DIR` outside the pinned tree;
4. verifies all process controls before importing Payne Zero;
5. verifies and reads the canonical raw subset, then independently derives the
   seven selected fields;
6. optionally re-extracts the full source row and checks the non-authoritative
   observed-row packing witness, without changing the physical payload;
7. builds the 80-depth fixed-state population and continuum threshold;
8. runs the full-array and projected-array selected-line calls as parity
   evidence, without treating either output as a published golden;
9. proves the projected fixture is sufficient and emits an in-memory mapping
   or temporary raw fixture capture plus a JSON summary;
10. has no publication option.

Run this worker twice in fresh processes before fixture acceptance. Require
identical fixture schemas, fixture payload/full fingerprints, full-array
hashes, projected arrays, and projection-parity outputs.

### 8.2 Oracle worker

After the fixture is canonical and manifest-bound, implement a separate
read-only worker such as:

```text
scripts/chapter06_atmosphere_oracle_worker.py
```

The oracle worker:

1. verifies the pinned source-code and line-table identities;
2. verifies the canonical fixture file, manifest entry, schema, every member
   hash, raw-subset identity, and conversion version before constructing the
   selected catalog;
3. loads only the fixture projections and reconstructs the three public-call
   arrays;
4. runs the public one-record selected-line call;
5. proves the one-line serial route and exact output contract;
6. constructs the downstream stimulated view and gate diagnostics;
7. emits an in-memory mapping or temporary raw oracle capture plus a JSON
   summary;
8. has no fixture-building or publication option.

Both workers require:

```text
LC_ALL=C
TZ=UTC
PYTHONHASHSEED=0
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
MKL_DYNAMIC=FALSE
MKL_NUM_THREADS=1
NUMBA_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMBA_CACHE_DIR=<fresh empty external temporary directory>
```

Reusing a populated cache directory must fail before importing the pinned
package. The atmosphere result is CPU-only NumPy/Numba; no GPU metadata belongs
in this golden.

### 8.3 Oracle double capture

Run two fresh oracle workers with two different empty Numba cache directories.
Require:

- identical key set and schema digest;
- identical oracle payload and full fingerprints, except an explicitly excluded
  temporary path token if paths are recorded canonically;
- identical canonical fixture identity;
- identical pre/post dense hashes and sparse encodings;
- identical selected count, gate mask, and nonzero support.

Any difference blocks candidate assembly. Do not convert a difference into a
tolerance without identifying its operation and owner.

### 8.4 Independent reviews

Acceptance is split:

1. **Conversion review:** raw row, formulas, seven values/dtypes, decoded
   quantities, and no cross-catalog claim.
2. **Fixture review:** source state, fixed-ne route, exact continuum
   recomputation, projection sufficiency, shapes/units/dtypes.
3. **Oracle review:** serial call chain, output, stimulation lifecycle,
   fingerprints, and seam evidence.
4. **Publisher review:** fail-closed identity checks and no-write behavior.
5. **Candidate artifact review:** rebuild the logical dense products,
   independently recompute all member hashes, and assess scientific
   suitability.

A reviewer must not accept its own artifact merely because its generating
tests pass.

## 9. Exact serial `njit` route

The required call chain is:

```text
accumulate_selected_line_opacity
  -> _accumulate_selected_line_opacity_compiled
  -> _accumulate_selected_line_wings_compiled
  -> _voigt_profile_compiled
```

with `_fast_exponential_lookup_compiled` before the wing call.

The public wrapper computes a possible chunk count, but invokes the parallel
wrapper only when both:

```text
chunk_count > 1
selected_lines.line_count > 1
```

The Chapter 6 record has `line_count == 1`, so the serial cached `njit` route
is mandatory even on a many-core machine. Tests should patch the parallel
wrapper to raise if called and show that the one-record public call still
succeeds. Do not create a Chapter 6 `prange` benchmark.

## 10. Required seam tests

The physical 80-depth fixture and the algorithmic seam cases have different
jobs. The physical fixture must not be tuned merely to hit every loop cap.
Small controlled in-memory arrays exercise those caps.

### 10.1 Eighty-depth requirement

Using otherwise valid inputs:

- `temperature.shape == (80,)` succeeds;
- 79 and 81 depths each raise
  `ValueError("Selected-line opacity expects ...")`;
- mismatched population/threshold depth axes fail before publication;
- the physical fixture is exactly 80 depths, outermost to innermost.

### 10.2 First red/blue ownership and 101/100 caps

Construct a monotonic 401-point float64 test grid with the reconstructed line
center exactly at index 200. Because the center walk advances while
`vacuum_wavelength_nm >= wavelength_grid[center_index]`:

- the first strictly red point is index 201;
- the first blue-side deposit is index 200.

Use:

- the one accepted record;
- all 80 depths;
- a broad positive controlled fractional width;
- positive line support at every depth;
- zero continuum threshold so no wing cutoff fires;
- sufficient grid extent on both sides.

Require exactly:

```text
red deposits:  indices 201..301 inclusive = 101
blue deposits: indices 200..101 inclusive = 100
total per active depth = 201
no deposit at index 100 or 302
```

This test owns the literal
`range(center_index, center_index + 101)` versus `range(1,101)` behavior.

### 10.3 First-below-cutoff inclusion

Use an independent scalar transcription with the exact sanitized float32
inputs to evaluate a monotonic run of wing contributions. Find adjacent points
for which a representable float32 threshold exists, then set

```text
q = nextafter(float32(contribution[j + 1]), +infinity)
```

and first require:

```text
contribution[j] >= q
contribution[j + 1] < q
```

Run the public one-line route and assert:

- point `j + 1` is present even though it is strictly below the cutoff;
- point `j + 2` is absent;
- perform the assertion independently on red and blue walks.

The threshold must remain below the center amplitude so the two center tests
do not suppress the line first.

Test wing equality separately against the compiled wing helper with controlled
Harris tables: Gaussian values exactly one, first/second corrections exactly
zero, offsets within ten Doppler widths, line-center absorption `1.0`, and
threshold `1.0`. Exact equality must continue to the applicable loop cap.
Changing the threshold to the next representable value above one must retain
the first deposited point and then stop. This isolates the literal post-deposit
`<` comparison without depending on an accidental equality in a physical
Voigt profile.

### 10.4 Exact 8-layer gates

Gate depths are one-based:

```text
8, 16, 24, 32, 40, 48, 56, 64, 72, 80
```

The intervening blocks are evaluated only when their bracketing gate state is
active. Use a positive controlled threshold and line-support patterns:

1. Put a strong line only at interior depth 10; leave gates 8 and 16 below the
   center cutoff. Depth 10 must remain zero and `selected_line_count` must be
   zero.
2. Activate gate 8 while keeping depth 10 strong. Gate 8 and depth 10 must now
   deposit and `selected_line_count` must be one.
3. Repeat with gate 16 instead of gate 8; depth 10 must again deposit.
4. Check the one-sided first block: depths 1–7 remain skipped when gate 8 is
   inactive and become eligible when gate 8 is active.
5. Check the final block with gates 72 and 80.

This proves the actual gate topology rather than merely asserting that the
fixture has 80 rows.

### 10.5 Center-cutoff ordering

At one gate depth, independently compute:

```text
pre_excitation_center_absorption
post_FASTEX_center_absorption
continuum threshold
```

Cover:

- pre-excitation below threshold -> skip;
- pre-excitation equal threshold -> continue;
- post-FASTEX below threshold -> skip;
- post-FASTEX equal threshold -> continue.

### 10.6 Dtype boundary

Require:

- `allocate_line_opacity_state(80,W)` returns a float64 zero slab;
- a nonempty selected-line call returns a float32 slab;
- the selected record retains `int32` wavelength and six `int16` fields;
- continuum threshold, electron density, line support, \(hc/kT\), and widths
  enter the compiled call through the declared float32 sanitization;
- actual packed populations and grid enter as float64;
- the downstream stimulated view is float64;
- no test converts the production line slab to float64 before checking its
  dtype and float32 bytes.

### 10.7 Projection and lifecycle seams

Require:

- full packed arrays and the published projections give bitwise-identical
  pre-stimulated output;
- perturbing any unowned packed column does not change the output;
- perturbing actual slots 0, 2, or 840 changes only the neutral-collision
  damping route;
- perturbing line-support or width slot 350 changes the line;
- `post == pre.astype(float64) * stimulated` on every cell;
- applying the stimulated factor twice fails;
- pre/post support is identical;
- the golden is not opened until the reader-built slab exists.

## 11. Feasibility witness from this design pass

A temporary read-only-source run used the construction above. It was not a
canonical double capture and the identities below are **not publication
authorization**. They are recorded so implementation begins from a checked
candidate rather than an untested idea.

### 11.1 Candidate fixture member byte hashes

| candidate array | C-byte SHA-256 |
| --- | --- |
| `temperature` | `7b76410e731a6772cb6de12f923aea8a26e345778dbab67db3e8938278ce270f` |
| `hc_over_kt` | `fdbd300b88163af886fe2905a883fa319c32abcc89db1099e4b0df7965d8174e` |
| runtime `electron_density` | `1e36bdf5aac263da2eac448f1b4c91e43e48a48afc133c8a8ec5e15735c85c18` |
| actual slots `[0,2,840]` | `ae17c6b96dc9eb824051707d83eca2bbdb22483af4c143b18d832326292e86ce` |
| Fe I line support slot 350 | `8625e8d942892d142384d51fc0625701374f79dfa471a78ff593d88479b3056f` |
| Fe I fractional width slot 350 | `f8018929e269db3e6b9a572d36b528bc0577a3a93bcc25166b277824ffb2e785` |
| 30,000-point wavelength grid | `8944f1dd701ba27f50d37a16e48ab9375e1bef1b444ed405b671ba91fde8132b` |
| 344 packed bin edges | `ae50e9c0bafdcbfc39e242fd5029bf53b9e712d43008c6cb78a4493b89272cd5` |
| `(80,344)` threshold | `76cb7ef18554149b61b97e32f2a69bbcdba3eb8b7e6b7da8ba3c69b8775b7293` |

Full temporary source-array fingerprints were:

```text
actual populations (80,1006)
  53ff8546517e744862abf16c490f1a3b4bcd761d681f5fde0cfb8ad62e0812b6
line support (80,1006)
  706c8db98522bacb0c428279a29b270ec53772174207cceadd27e701947030c0
fractional widths (80,1006)
  23c57f545bdfd74af3dd62ef57b8f347570953ea50c4e19289a9aa9715beb868
```

### 11.2 Candidate output

The line center mapped to:

```text
reconstructed wavelength      499.03410878793585 nm
first red grid index          7383, zero based
neighbor grid values          498.99937308, 499.11428517 nm
continuum reference column    169, zero based
```

The temporary output had:

```text
shape                         (80,30000)
dtype                         float32
selected_line_count           1
nonzero values                240
nonzero values per depth      3
active 8-layer gates          all ten
peak pre-stimulated opacity   0.3731258809566498 cm^2 g^-1
pre-stimulated dense hash     43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58
post-stimulated dense hash    a695164eea303f76f249b1a81ce179426d16e9047cefa079b09a9ff3cfae35ab
sparse index hash             9ff0b9443c843eb582ef7d69a1d46ffab6690d938cd213215e6affda680a40f4
sparse pre-value hash         8ad01776bee089b24f3190f45e1bcaee106e7438b7c804c1c263bb78c8fac040
sparse post-value hash        669e5c9928edb9ed7f05b1711b2661e1d4fdfd5b91b06508c94b78745fe996eb
```

The three-pixel-per-depth result is expected on the atmosphere's direct coarse
30,000-point sampling grid near 499 nm. It is suitable for exact integration
parity, but not for teaching profile normalization or for proving the
101/100-point cap. Those claims remain assigned to the dense teaching profile
and controlled seam tests, respectively.

The implementation gate may promote these hashes to accepted identities only
after two fresh fixture captures agree, the fixture is independently accepted
and published, and two fresh oracle captures against that exact fixture also
agree. A changed worker or adapter invalidates the corresponding identities.

## 12. Fail-closed publication gates

The publisher should have no implicit “best available” mode and no force
overwrite option.

Proposed reviewed entry points are:

```text
scripts/build_chapter06_atmosphere_fixture.py
scripts/build_chapter06_atmosphere_golden.py
design/chapter06_atmosphere_fixture_publication_acceptance.json
design/chapter06_atmosphere_oracle_publication_acceptance.json
```

The two scripts are deterministic publishers, not scientific workers. Each
must reauthorize its own detached record and canonical destination immediately
before the no-replace rename.

The raw subset is already a canonical manifest-bound input. Do not republish it
as part of the atmosphere operation. Publication has two ordered, separately
reviewed phases:

1. publish and manifest-register the deterministic atmosphere **fixture**;
2. capture against that exact canonical fixture, then publish and
   manifest-register the comparison-only atmosphere **golden**.

This ordering is intentional. It makes the golden depend on an immutable input
hash and avoids pretending that files in separate role directories can become
visible through one atomic filesystem rename.

Before fixture publication, require:

1. exact source commit and all source/data hashes;
2. exact accepted worker, converter, publisher, contract, and acceptance-record
   hashes;
3. canonical raw-subset hash, builder hash, schema, manifest entry, raw row
   873702 identity, and all seventeen field dtypes/values;
4. exact seven selected values, dtypes, member hashes, four-word payload, and
   decoded physical ledger;
5. exact 80-depth source file and adapter configuration;
6. exact full-array source fingerprints and projection equality;
7. fixture candidate scientific review and deterministic byte reproduction;
8. the canonical fixture destination is absent, or already contains exactly
   the accepted bytes;
9. the destination resolves inside `data/fixtures`, with no symlinked path
   component.

Publish the fixture with an atomic same-filesystem no-replace **file** rename,
then atomically replace `data/MANIFEST.json` with the accepted fixture entry.
Readers must reject an unregistered file even if it is present. A crash after
the file rename but before manifest replacement therefore leaves an inert
candidate: a resumed publisher may continue only if its bytes equal the
detached acceptance record exactly.

Before golden publication, require all fixture gates again and additionally:

1. the fixture exists at its canonical path and its manifest member metadata
   and bytes match the accepted fixture;
2. exact two-capture schema and oracle-payload/full fingerprints;
3. line count one, serial route, selected count one;
4. output shape `(80,30000)`, pre dtype float32, post dtype float64;
5. exact sparse reconstruction and dense hashes;
6. ten valid gate decisions and all required seam tests;
7. the golden candidate passes independent scientific and artifact review;
8. the canonical atmosphere golden file is absent, or already contains exactly
   the accepted bytes;
9. the destination resolves inside `data/golden/payne_zero`, with no symlinked
   path component.

Stage the golden file on the same filesystem, verify its final bytes, and use
an atomic no-replace **file** rename. This atmosphere file must remain
independently publishable from the separately owned synthesis golden; the
publisher must not claim ownership of the whole Chapter 6 directory. Then
atomically replace `data/MANIFEST.json` with the accepted golden entry. A
present but unregistered exact file is inert and may be recovered only through
the same acceptance checks; any nonexact partial state is a hard failure.

After each phase, run a post-publication manifest audit from a separate
process. Register Chapter 6 as available only after both phases and the
executed notebook/HTML pass. Retain detached acceptance records as immutable
history; do not rewrite them to match later files.

Mutation tests must prove no canonical writes for:

- one changed source hash or commit;
- any changed selected value/dtype;
- swapped selected halfwords;
- a different source row;
- a source shift/correction or default-damping branch;
- 79/81 depths;
- a changed fixture projection;
- a changed full source fingerprint;
- a changed process control or reused Numba cache;
- a changed output support/value/dtype/hash;
- missing or extra artifact member;
- a stale or malformed authorization record;
- a symlinked or out-of-tree destination;
- an already-present nonexact or unauthorized partial publication;
- a fixture that exists but is absent from the accepted manifest when golden
  capture begins.

## 13. Blockers and ambiguities

### Not blockers if labeled correctly

1. **The structured solar example lacks full modern provenance and convergence
   metadata.** It is adequate as an upstream integration state, not as a claim
   of newly established atmosphere convergence. The exact fixed-state
   recomputation and all source hashes remain explicit.
2. **The observed atmosphere row matches the converted payload.** That is
   packing corroboration, not catalog genealogy. It must remain outside the
   fixture authority chain.
3. **The native line occupies only three atmosphere pixels per depth.** The
   candidate is still visible at every depth and every gate. Profile
   normalization and wing-cap pedagogy use their separately owned grids.
4. **The published fixture is a projection.** Full-versus-projected bitwise
   parity is a required gate, and the member names say “slot values,” not a
   falsely complete packed atmosphere.

### Must be closed before publication

1. The exact fixture builder, oracle worker, and publishers do not exist yet.
2. The fixture and oracle feasibility hashes each need two fresh-process
   reproduction runs and independent acceptance.
3. The sparse golden encoding needs a tested canonical reconstruction helper
   and deterministic writer.
4. The final dynamic read-set ledger must show no undeclared source inputs.
5. The source example's missing constructor fields must be proven unread by
   the fixed-state fixture route through explicit perturbation tests.

If review rejects the packaged structured example as a fixture source, the
safe fallback is a fresh pinned solver run that writes an internal debug state
to a temporary directory outside the source tree. That is a new capture
contract and requires new fingerprints; it must not silently replace the
source described here.

## 14. Safe implementation sequence

1. Reverify the canonical teaching-subset hash, schema, builder, manifest
   entry, and all 17 raw fields; freeze the atmosphere conversion version.
2. Implement a pure converter with no output path and exhaustive row/quantizer
   tests.
3. Independently reproduce the seven values and four-word payload; retain the
   observed row only as non-authoritative corroboration.
4. Implement the fixture adapter and fixed-state capture in memory.
5. Verify the complete dynamic source read set and pin every identity.
6. Prove placeholder-field non-ownership and full-versus-projected output
   identity.
7. Implement the serial atmosphere oracle and exact pre/post stimulation
   ledger.
8. Implement the 80-depth, 101/100, 8-layer, first-below-cutoff, center
   equality, and dtype seam tests.
9. Run two fresh empty-cache fixture captures, then independently review,
   publish, and manifest-register the deterministic fixture.
10. Run two fresh empty-cache captures against that exact canonical fixture
    under the fixed environment.
11. Freeze the accepted oracle schema and oracle-payload/full fingerprints only
    after independent review.
12. Implement and adversarially test a deterministic no-overwrite golden
    publisher.
13. Assemble the golden candidate in a temporary directory; independently
    audit its science, provenance, sparsity, and reconstruction.
14. Create a detached exact authorization record.
15. Atomically publish the accepted atmosphere golden file without
    replacement.
16. Synchronize and independently audit `data/MANIFEST.json`.
17. Only then let the Chapter 6 notebook load the fixture and, after its own
    calculation, open the golden.

This order keeps source evidence, physical input, and comparison output
separate; prevents a lossy packed match from becoming a false cross-catalog
claim; and gives Chapter 6 an exact serial atmosphere result without importing
Chapter 7's catalog-selection or parallel-forest responsibilities.
