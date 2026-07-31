# Chapter 6 atmosphere-converter independent audit

Status: independent candidate review  
Disposition: **ACCEPT for safe-sequence steps 1–3**  
Audited: 2026-07-30

## 1. Scope and immutable review snapshot

This audit reviewed, without editing the converter or its tests:

- `scripts/chapter06_atmosphere_line_converter.py`, 619 lines, SHA-256
  `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb`;
- `tests/test_chapter06_atmosphere_line_converter.py`, 585 lines, SHA-256
  `254d796b7ab761ca806c372d0bcdd935067ff1a89b2acfebcfa3007fe3f549dc`;
- `design/chapter06_atmosphere_converter_candidate.md`, SHA-256
  `a973f377944c808a7bb4845828fc8936d1b2d31170474dc65a1904ceb5d119f4`;
- `design/chapter06_atmosphere_fixture_oracle_plan.md`, SHA-256
  `cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97`;
- the canonical raw subset, SHA-256
  `bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04`;
- the current `data/MANIFEST.json`, SHA-256
  `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a`;
- `scripts/build_chapter06_fe_record_subset.py`, SHA-256
  `25bcf4662740155e8b08615b9522f3f4517e1a5ddc4627c68686620ccfff4d6c`.

The candidate narrative predates the localized provenance repair and still
records the predecessor converter/test hashes. It is reviewed as design
context, not as identity authority for the accepted repaired snapshot; this
independent report owns the accepted hashes above.

The pinned Payne Zero and paper trees remained read-only.

Commands and outcomes:

```text
python -m pytest -q \
  tests/test_chapter06_atmosphere_line_converter.py \
  tests/test_chapter06_fe_record_subset.py \
  tests/test_chapter06_source_data.py
28 passed in 1.82s

ruff check \
  scripts/chapter06_atmosphere_line_converter.py \
  tests/test_chapter06_atmosphere_line_converter.py
All checks passed!

ruff format --check \
  scripts/chapter06_atmosphere_line_converter.py \
  tests/test_chapter06_atmosphere_line_converter.py
2 files already formatted
```

This report is the only file modified by the audit.

## 2. Executive verdict

The candidate's numerical conversion is correct. Independent calculations
reproduced:

- all seventeen raw fields and their fixed-width byte semantics;
- the exact quantizer formulas and storage domains;
- Fe I packed population slot `3510`;
- all seven selected-record values, dtypes, and C-byte hashes;
- native little-endian halfword ordering;
- the four-word payload and no-swap round trip;
- the exact float32 kernel-facing physical ledger;
- staged source-module origin;
- the absence of output, publication, optional-catalog, or golden interfaces;
- the deliberately non-authoritative status of observed row `780108`.

The provenance boundary is now repaired. The public canonical loader:

- has no path argument;
- reads only the fixed canonical subset;
- rejects a symlinked subset, manifest, or builder;
- requires a unique canonical manifest entry;
- verifies manifest schema and pinned commit;
- verifies the exact file, source archive, row, schema, role, format, and
  builder identities;
- hashes the actual builder file;
- checks all 27 archive members against manifest shapes, normalized dtypes,
  and C-byte hashes.

Independent adversarial checks rejected an alternate same-byte copy, a
symlinked canonical path, duplicate manifest JSON keys, a duplicate target
entry, changed top-level authority, changed member metadata, and a changed
builder file.

The disposition is therefore **ACCEPT for safe-sequence steps 1–3**. This
acceptance covers only the immutable converter/test hashes in Section 1. It
does not authorize an atmosphere fixture, oracle, golden, or publication.

## 3. Repaired provenance boundary

### P0-1 — The public loader reads only the canonical artifact: PASS

The repaired public signature is:

```python
def load_verified_canonical_raw_row() -> dict[str, np.ndarray]:
```

An exact-byte copy cannot be passed to the public loader:

```python
with tempfile.TemporaryDirectory() as directory:
    alternate = Path(directory) / "same-bytes-unregistered.npz"
    alternate.write_bytes(CANONICAL_SUBSET_PATH.read_bytes())
    load_verified_canonical_raw_row(alternate)  # TypeError
```

The lower-level alternate-path archive validator is now private and is used
only for adversarial byte/schema tests. The public loader also rejects when
its fixed canonical subset path is a symlink. This proves that same bytes at
an unregistered location cannot silently become canonical input.

### P0-2 — Manifest and builder authority are execution gates: PASS

Loader success is now conditional on `_validate_manifest_authority`. It
requires:

```text
top-level manifest schema_version = 1
top-level Payne Zero commit        = 9c44001feae40b85146630499e6f8a5fed42e5af
exactly one canonical path entry
role / format                      = subset / npz
subset bytes / SHA                 = 8665 / bb7ae01...
source archive path / bytes / rows / SHA
source row / field count / subset schema
builder path / builder SHA
requires_optional_full_catalog     = true
```

The actual builder file is a regular nonsymlink file and hashes to
`25bcf4662740155e8b08615b9522f3f4517e1a5ddc4627c68686620ccfff4d6c`.

The manifest member set must exactly equal the loaded 27-member archive set.
For every member, the gate checks shape, a normalized NumPy dtype, and the
canonical C-byte SHA. Duplicate JSON keys are rejected while loading the
manifest, and duplicate target entries are rejected separately.

Independent mutations of the root commit, duplicate target entry, member
hash, builder bytes, and duplicate JSON keys all failed closed.

## 4. Accepted numerical and semantic evidence

### 4.1 All seventeen raw fields: PASS

The candidate requires the exact set and order of seventeen raw names, shape
`(1,)`, and the frozen dtypes:

```text
11 float64
4 int64
1 |S10
1 |S3
```

Every scalar value matches Section 4.1 of the plan. The blank shift field
retains ten space bytes; the category tag retains three zero bytes. Missing,
extra, shape-changed, dtype-changed, and value-changed fields are rejected.

The archive itself is required to have exactly the sorted 27-member schema,
size `8665`, and the canonical SHA-256 before its members are used.

### 4.2 Quantizer formulas, rounding, and domains: PASS

The implementations match the frozen formulas:

```python
floor(log(wavelength_nm) / ratio_log_step + 0.5)
floor(log10(positive_value) * 1000.0 + 16384.5)
```

Both reject NaN, infinities, zero, and negative inputs. Wavelength storage is
range-checked before the `int32` cast. TABLOG accepts only codes `1..32767`
and rejects conceptual codes `0` and `32768`, preventing signed-`int16`
wrapping.

Independent controlled one-sided rounding checks gave:

```text
TABLOG unrounded index 20000.5-eps -> 20000
TABLOG unrounded index 20000.5     -> 20001
TABLOG unrounded index 20000.5+eps -> 20001

wavelength unrounded index 12425352.5-eps -> 12425352
wavelength unrounded index 12425352.5     -> 12425353
wavelength unrounded index 12425352.5+eps -> 12425353
```

The focused tests should eventually pin these one-sided half-step cases
directly; the current implementation itself is correct.

### 4.3 Species slot and seven selected fields: PASS

The converter uses staged
`atomic_population_slot_start(26) == 350`, adds one for Fe I, and obtains:

```text
population slot one based = 351
packed species slot       = 3510
abs(3510) // 10           = 351
```

Independent formulas reproduced:

| member | value | dtype | C-byte SHA-256 |
| --- | ---: | --- | --- |
| `packed_wavelength_index` | `12425352` | `int32` | `c423f4ac4a3825c6fad5336a1e15c0038ab17087f930916ad550ad45b4990dfc` |
| `packed_species_slot` | `3510` | `int16` | `c2126161f7488b7d198ea310da9a8694786a18a2317eea5e30379ee118d34743` |
| `lower_excitation_index` | `20909` | `int16` | `9b5bf0b74e2e212b57bcb2b9f2712eaab4c6169595f4e82767793f4534365648` |
| `log_strength_index` | `15524` | `int16` | `3fe821d54660a0c51b42d19a42571e208791b3c5ff6bc0cac16b6553a226515a` |
| `radiative_damping_index` | `24854` | `int16` | `5cdf3b75730a5b45c3f24da2c1030103143981191d2844aa7374e948b9abaeea` |
| `stark_damping_index` | `11934` | `int16` | `91d82e7b89ae43b29bb673bb416697d005e093d0011c9d7af501630c6a502141` |
| `van_der_waals_damping_index` | `8874` | `int16` | `c0346f09a0362e8e9d29c01a9fc7c292a7395b524a90319bb921608b9fdb1b60` |

The damping codes come from the three raw positive damping values, not from
the synthesis-normalized fields.

### 4.4 Native halfwords and no-swap decoding: PASS

On the declared native little-endian host, independently packing the six
`int16` halfwords in field order produced:

```text
[[12425352, 1370295734, 1628847268, 581578398]]
```

The payload has shape `(1,4)`, dtype `int32`, and SHA-256
`1769c9ad8d33e847a099bd6d50df85a2f478f98d554b2fc39db8121ba93158d2`.

`decode_selected_line_words(..., detect_swapped_layout=False)` reconstructed
all seven arrays, dtypes, and bytes exactly. The packer rejects silent member
shape and dtype casts.

### 4.5 Float32 physical ledger: PASS

The decoded lookup and literal kernel ordering reproduce:

```text
reconstructed wavelength       499.03410878793585 nm
unquantized wavelength         499.03411946178176 nm
difference                    -1.0673845906694623e-05 nm
lower excitation               33496.54296875 cm^-1
gf                              0.13803842663764954
raw radiative gamma             295120928.0 s^-1
raw Stark coefficient           3.548133827280253e-05 cm3 s^-1
raw van der Waals coefficient   3.090295308538771e-08 cm3 s^-1
classical strength              3.440358938918034e-18 cm2
normalized radiative damping    3.909297063842132e-08
normalized Stark damping        4.700008332680409e-21 cm3
normalized van der Waals        4.093536498989339e-24 cm3
```

These values correctly remain distinct from the readable synthesis record.
No atmosphere/synthesis slab equality or catalog-row genealogy is claimed.

### 4.6 Source isolation: PASS

The two imported atmosphere modules resolved to:

```text
src/payne_zero_atmosphere/line_catalog.py
src/payne_zero_atmosphere/population_layout.py
```

Their hashes are respectively:

```text
2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92
36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0
```

Both are byte-identical to the pinned external Payne Zero checkout. The
converter rejects an already-imported atmosphere module outside the staged
source root and puts the staged root first before importing its dependencies.

### 4.7 No output/publication surface and no false identity: PASS

The module has no output path, CLI, `save`, `savez`, `tofile`, fixture,
golden, or publication function. Its public loader reads only the fixed
canonical subset plus the manifest and builder identities needed to authorize
that input.

`NON_AUTHORITATIVE_OBSERVED_ROW_INDEX = 780108` is explicitly named as
corroboration. The module contains no observed-catalog path or loader and did
not read `observed_atomic_lines.npy`. It does not claim that synthesis row
`873702` and atmosphere row `780108` share upstream identity.

## 5. Acceptance scope

The converter is **ACCEPTED for safe-sequence steps 1–3** at the immutable
converter/test hashes in Section 1. The 28 focused source/converter tests,
independent canonical-authority adversarial checks, independent seven-code
and native-payload reconstruction, float32 physical ledger, staged source
identity, and lint/format checks are green.

Steps 4 and later remain outside this converter audit. No fixture, oracle,
golden, or publication is authorized by this report.
