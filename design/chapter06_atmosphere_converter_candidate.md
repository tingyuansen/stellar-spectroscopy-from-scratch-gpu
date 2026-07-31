# Chapter 6 atmosphere converter candidate

Status: **independently accepted for safe-sequence steps 1–3 only**; this
acceptance authorizes the repaired converter as a fixture-worker dependency,
not any fixture, oracle, golden, or publication

Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

## Scope

This pass implemented one pure, provenance-bound conversion:

```text
accepted raw synthesis-source row 873702
  -> seven one-record SelectedLineCatalog arrays
  -> native four-word selected-line payload
  -> decoded atmosphere-kernel physical ledger
```

The converter has no output path. It does not build an 80-depth atmosphere,
open an atmosphere source structure, run line opacity, serialize a fixture,
read or write a golden, or modify the pinned Payne Zero and paper trees.

The implementation is deliberately specific to the accepted ordinary,
zero-shift, correction-free Fe I teaching row. It is not a general converter
for arbitrary damping defaults, isotopic shifts, line categories, or species.

## Accepted implementation identities

| file | SHA-256 |
| --- | --- |
| `scripts/chapter06_atmosphere_line_converter.py` | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| `tests/test_chapter06_atmosphere_line_converter.py` | `254d796b7ab761ca806c372d0bcdd935067ff1a89b2acfebcfa3007fe3f549dc` |
| `design/chapter06_atmosphere_converter_independent_audit.md` | `60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75` |
| staged `src/payne_zero_atmosphere/line_catalog.py` | `2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92` |
| staged `src/payne_zero_atmosphere/population_layout.py` | `36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0` |

The candidate conversion version is `1`.

## Step 1: canonical raw authority

The loader checks the canonical subset before returning a defensive copy:

- exact file size `8665` bytes;
- exact SHA-256
  `bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04`;
- exact sorted 27-member archive schema;
- exact schema version `1`, source row `873702`, source archive identity,
  source commit, field count, and static-role declaration;
- exact shape `(1,)`, dtype, and scalar value of all 17 raw fields;
- ten literal space bytes in `energy_shift_field`;
- three literal zero bytes in `line_category_tag`.

Mutation tests reject missing and extra members and, independently for each of
the 17 raw fields, a changed shape, dtype, or value. The converter repeats this
validation before performing any conversion math.

## Step 2: pure fail-closed conversion

The two quantizers use the formulas and storage domains frozen by the
atmosphere fixture plan:

```text
packed wavelength: floor(log(lambda_nm) / log(1 + 1/2,000,000) + 0.5)
TABLOG:            floor(1000 log10(value) + 16384.5)
```

Both reject nonfinite and nonpositive input. The wavelength code is range
checked before the `int32` cast. TABLOG accepts only `1..32767`; it never
wraps conceptual entry 32768 into negative `int16`.

The record conversion uses the energy-derived wavelength, the smaller
absolute energy as the lower excitation, both zero oscillator-strength
corrections, and all three explicit raw positive damping quantities. It does
not reuse the synthesis-normalized damping fields. The Fe I population
boundary is derived through the staged packed population layout:

```text
atomic_population_slot_start(26) = 350  # zero based
Fe I population slot             = 351  # one based
ordinary packed species slot     = 3510
```

## Step 3: independently reproduced packed result

An independent scalar transcription in the focused test reproduces all seven
codes without calling the converter's quantizer helpers. It separately packs
the six halfwords and obtains the same four-word payload.

| selected member | value | dtype | C-byte SHA-256 |
| --- | ---: | --- | --- |
| `packed_wavelength_index` | `12425352` | `(1,) int32` | `c423f4ac4a3825c6fad5336a1e15c0038ab17087f930916ad550ad45b4990dfc` |
| `packed_species_slot` | `3510` | `(1,) int16` | `c2126161f7488b7d198ea310da9a8694786a18a2317eea5e30379ee118d34743` |
| `lower_excitation_index` | `20909` | `(1,) int16` | `9b5bf0b74e2e212b57bcb2b9f2712eaab4c6169595f4e82767793f4534365648` |
| `log_strength_index` | `15524` | `(1,) int16` | `3fe821d54660a0c51b42d19a42571e208791b3c5ff6bc0cac16b6553a226515a` |
| `radiative_damping_index` | `24854` | `(1,) int16` | `5cdf3b75730a5b45c3f24da2c1030103143981191d2844aa7374e948b9abaeea` |
| `stark_damping_index` | `11934` | `(1,) int16` | `91d82e7b89ae43b29bb673bb416697d005e093d0011c9d7af501630c6a502141` |
| `van_der_waals_damping_index` | `8874` | `(1,) int16` | `c0346f09a0362e8e9d29c01a9fc7c292a7395b524a90319bb921608b9fdb1b60` |

The native payload is:

```text
[[12425352, 1370295734, 1628847268, 581578398]]
```

Its shape is `(1,4)`, dtype is `int32`, and C-byte SHA-256 is
`1769c9ad8d33e847a099bd6d50df85a2f478f98d554b2fc39db8121ba93158d2`.
Decoding with `detect_swapped_layout=False` reproduces the seven input arrays,
dtypes, and bytes exactly.

## Decoded physical ledger

The candidate ledger reconstructs the atmosphere lookup inputs in their
literal float32 production order:

| quantity | decoded value |
| --- | ---: |
| wavelength | `499.03410878793585 nm` |
| unquantized energy-derived wavelength | `499.03411946178176 nm` |
| reconstructed minus unquantized | `-1.0673845906694623e-05 nm` |
| lower excitation | `33496.54296875 cm^-1` |
| oscillator strength | `0.13803842663764954` |
| raw radiative damping | `295120928.0 s^-1` |
| raw Stark coefficient | `3.548133827280253e-05 cm^3 s^-1` |
| raw van der Waals coefficient | `3.090295308538771e-08 cm^3 s^-1` |
| classical strength | `3.440358938918034e-18 cm^2` |
| normalized radiative damping | `3.909297063842132e-08` |
| normalized Stark damping | `4.700008332680409e-21 cm^3` |
| normalized van der Waals damping | `4.093536498989339e-24 cm^3` |

The float32 ledger is intentionally not identified with the readable
synthesis record. Packing quantizes wavelength, excitation, strength, and all
three damping inputs.

## Verification

Executed:

```text
ruff format scripts/chapter06_atmosphere_line_converter.py \
  tests/test_chapter06_atmosphere_line_converter.py

ruff check scripts/chapter06_atmosphere_line_converter.py \
  tests/test_chapter06_atmosphere_line_converter.py

PYTHONPATH=src:. pytest -q \
  tests/test_chapter06_atmosphere_line_converter.py \
  tests/test_chapter06_fe_record_subset.py \
  tests/test_chapter06_source_data.py
```

Result:

```text
Ruff: clean
28 passed
```

The focused converter tests cover:

- canonical file, manifest, schema, provenance, and all 17 raw fields;
- file-byte, field-set, shape, dtype, and value mutation failures;
- nonfinite, nonpositive, lower-index, upper-index, and storage-width
  quantizer failures;
- exact seven values, dtypes, and member hashes;
- independent seven-code and four-word reproduction;
- no-swap decode round-trip;
- packer refusal to silently cast member shapes or dtypes;
- exact decoded float32 physical ledger.

## Non-authoritative corroboration

The implementation records `780108` only as the documented optional observed
row index. It contains no path or loader for `observed_atomic_lines.npy`, and
the optional observed catalog was not read during this pass. Therefore the
observed match cannot enter the converter's authority chain or become a
reader dependency.

## Independent acceptance

The repaired immutable converter and test identities above were independently
reviewed and accepted in
`design/chapter06_atmosphere_converter_independent_audit.md`. That audit
reproduced the formulas, quantizer domains, seven members, native halfword
order, decoded ledger, manifest-bound canonical loader, staged source origin,
and absence of any cross-catalog identity claim.

The acceptance closes safe-sequence steps 1–3 and permits a later in-memory
fixture worker to pin this exact converter. Steps 4 and later remain separate
scientific and publication gates; no fixture, oracle, golden, or canonical
write is authorized by this record.
