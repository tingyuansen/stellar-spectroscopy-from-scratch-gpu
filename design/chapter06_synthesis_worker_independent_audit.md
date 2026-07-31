# Chapter 6 synthesis scientific-worker independent audit

Status: **FINAL ACCEPT — frozen scientific worker; publication remains unauthorized**  
Scope: six-depth CPU synthesis scientific worker only  
Audited and re-audited on: 2026-07-30  
Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

This is an independent review of:

- `scripts/chapter06_synthesis_oracle_worker.py`;
- `tests/test_chapter06_synthesis_oracle_worker.py`;
- `design/chapter06_synthesis_fixture_oracle_plan.md`;
- `design/chapter06_exact_source_contract.md`.

The worker and its tests were not edited by this review. No raw capture,
compact candidate, golden, manifest entry, publication authorization, or
external Payne Zero source was written.

## 1. Disposition

**FINAL ACCEPT the frozen 754-member scientific worker.**

The initial independent review rejected worker
`e127521d61ff185cfc12cfc28b9a13639b96345e86a1b44453460e2f849d9281`
because it omitted the full stimulated-emission arrays and accepted symlink
aliases to canonical inputs. Both findings are closed in the repaired worker:

1. all eight full `(depth=6, wavelength=W)` `float32` stimulated-emission
   factors are returned, bounded, and used to reconstruct both net routes
   bitwise;
2. fixture and subset symlink aliases and ordinary alternate paths fail before
   `np.load`.

The exact one-record construction, continuum recomputation, source isolation,
dtype fences, activity masks, reach accounting, loop/batched parity, and
fresh-process determinism also pass.

The independently authorized constants are now frozen as:

```text
ACCEPTED_CAPTURE_KEY_COUNT = 754
ACCEPTED_CAPTURE_SCHEMA_DIGEST =
    d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178
```

Two final fresh-process runs prove that the frozen worker sets
`meta__capture_scope_complete=True`, preserves the authorized schema and
physical payload, and produces the expected new full fingerprint that binds
the frozen worker and acceptance metadata.

This final scientific-worker acceptance authorizes the separate
frozen-worker acceptance record. It does not authorize serialization,
candidate publication, manifest mutation, or golden publication.

## 2. Closure of the prior blocking findings

### P0 closed — all full stimulated-emission factors are retained

For every Cartesian product of

```text
regime = hot_dwarf, solar_dwarf, low_gravity_giant, cool_molecule_rich
grid   = canonical, coarse
```

the repaired raw mapping contains exactly one member named

```text
{regime}__{grid}__ledger__stimulated_emission_factor_float32
```

The eight members have:

- shape `(6,6000)` on the canonical grid or `(6,400)` on the coarse grid;
- NumPy dtype `float32`;
- C-contiguous storage;
- finite values strictly greater than zero and no greater than one;
- global minimum `0.5894814729690552`;
- global maximum `0.9999690055847168`.

For each regime, grid, and wing mode, independent array checks proved exact
equality:

```python
np.multiply(gross, stimulated_factor, dtype=np.float32) == net
```

This holds for all eight batched/loop route pairs. Every full factor's center
column is also bitwise equal to the corresponding
`stimulated_center_factor_float32` ledger member.

The raw capture now satisfies the design requirement to retain the full
factor before a future publisher reduces it to approved digests and selected
samples.

### P1 closed — canonical inputs fail closed

The repaired `_canonical_input`:

1. checks both absolute and resolved paths against the golden root;
2. rejects a supplied symlink leaf before reading it;
3. compares the supplied absolute path with the exact canonical input path;
4. verifies the canonical target is a regular nonsymlink file.

Independent negative probes covered:

- a fixture symlink alias;
- a subset symlink alias;
- an ordinary alternate fixture path;
- an ordinary alternate subset path.

All four raised `OracleIdentityError`. The focused test patches `np.load` to
raise if reached, proving the failures occur before any archive read.

## 3. Exact identities reviewed

| artifact | SHA-256 |
| --- | --- |
| frozen synthesis worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| frozen focused tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| synthesis fixture/oracle plan | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` |
| exact source contract | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |
| Chapter 5 state fixture | `ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae` |
| one-row Fe I subset | `bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04` |
| synthesis continuum tables | `406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d` |
| synthesis continuum edge grid | `11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e` |
| synthesis line-profile tables | `87b47fc76bed10455218f43c4b6686525b961002e72d6a5ef01255a08deb27d4` |
| full source archive | `4eafa927c02a4f74401523149a44e35239f2aaecb4a64f2905a4cd5530c2dde7` |

The worker verified the exact 17-file loaded pinned-Python manifest before and
after the science route. The five staged sources executed directly by the
worker were byte-identical to the pinned checkout. All three staged static
archives were byte-identical to their upstream authorities.

## 4. Complete raw-schema review

The repaired in-memory result has **754 object-free members** and
**8,451,402 array bytes**.

| prefix/family | member count | reviewed content |
| --- | ---: | --- |
| `meta__` | 57 | process, source-set, contract, dtype, and scope identities |
| `identity__` | 27 | pinned code, archive, and table hashes |
| `fixture_payload__` | 2 | exact field inventory and fixture digest |
| `subset_payload__` | 2 | exact field inventory and subset digest |
| `record__` | 14 | 13 derived fields plus the field-name inventory |
| `catalog__` | 18 | exact 13 physical plus five support entries |
| `grid__` | 8 | both axes, hashes, and center/wing indices |
| `invariant__` | 86 | all 43 invariant fields on both grids |
| four regime families | 540 | 135 members per regime |

The eight added members are exactly the eight full stimulated-emission
factors. No other schema drift was observed.

The frozen raw-capture identities are:

| property | exact value |
| --- | --- |
| schema version | `1` |
| key count | `754` |
| array bytes | `8,451,402` |
| schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical-payload fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| full-capture fingerprint | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` |

The key count and schema digest exactly match the prior independent
authorization. The physical fingerprint is unchanged from the pre-freeze
candidate, proving that freezing altered no scientific payload. The full
fingerprint changed from
`2005cb7f1794f66ed04e1f90858c77430c8b1693048ca326a49b0473af6ff194`
to the value above, as expected because it binds the worker hash and accepted
scope metadata.

## 5. Positive scientific findings

### Source and data isolation

- The exact external commit, pinned module hashes, loaded module set, staged
  source hashes, static table hashes, full source-archive identity, fixture
  identity, and subset identity are checked before importing Payne Zero.
- The fixture has the exact 217-member schema and payload digest.
- The subset has 27 members: 17 exact raw fields and ten provenance fields,
  with no object dtype.
- Golden paths are rejected before `np.load`.
- The worker exposes no output, destination, serialization, publication, or
  golden argument.
- `meta__golden_read_performed` and
  `meta__golden_publication_performed` are both false.
- The independently supplied, pre-created disposable Numba cache directories
  remained empty. No project, source, paper, raw-capture, or golden file was
  written.

### Exact one-record mapping and invariants

- `_build_records` derives the accepted Fe I row exactly.
- The constructor mapping contains 13 physical per-line fields and all five
  required support entries.
- The ordinary line count is one.
- Fe I population indices are exactly stage `0`, element `25`.
- All 11 autoionizing arrays and all ten helium array fields exist with exact
  zero lengths and dtypes.
- Harris arrays are `(2001,) float64`; FASTEX arrays are `(1001,) float64`.
- Canonical CPU work is Torch `float64`; invariant strength/damping uploads,
  cutoff inputs, stimulation, and accumulation are Torch `float32`.

### Four regimes, grids, and line lifecycle

- Canonical grid: 6000 `float64` samples, center and wing index `2434`,
  mapped center wavelength `499.0333758196059` nm.
- Coarse grid: 400 `float64` samples, mapped center wavelength
  `499.04420746429804` nm.
- Both grids reproduce:

```text
hot_dwarf            1 1 1 0 0 0
solar_dwarf          1 1 1 1 1 1
low_gravity_giant    1 1 1 1 1 1
cool_molecule_rich   1 1 1 1 1 1
```

- Active canonical wing reach spans 5–163 samples.
- For every regime and grid, gross/net and batched/loop slabs are
  `(6,W) float32`, finite, and nonnegative.
- Gross loop and batched slabs are bitwise equal.
- Net loop and batched slabs are bitwise equal.
- Both net modes reconstruct exactly from their gross slab and the retained
  full stimulated factor.
- Activity, center deposition, wing reach, and nonzero counts reconstruct one
  another; active counts satisfy `2 * reach + 1`.
- The canonical maximum gross and net cells are respectively
  `4.732271671295166` and `4.676202774047852`
  cm2 g-1.
- Electron-density and neutral-collision perturbation ledgers isolate Stark
  and van der Waals damping respectively.

## 6. Fresh-process determinism

Two final independently launched processes used unrelated, empty external
cache directories and the exact one-thread/C-locale environment. Their
complete summaries agreed on:

- `capture_scope_complete=True`;
- scope text
  `accepted exhaustive Chapter 6 CPU one-line synthesis capture`;
- all frozen identities in Section 4;
- eight full stimulated-factor members and their exact extrema;
- exact batched and loop net reconstruction;
- both four-regime activity masks;
- canonical reach 5–163;
- zero loop/batched absolute difference;
- 17 loaded pinned source files;
- no golden read or publication.

The test compares summaries rather than serialized raw bytes. Both
fingerprints cover the complete object-free array mapping, so this is
sufficient for final scientific-worker acceptance. Deterministic raw NPZ byte
identity remains a publisher-stage gate.

## 7. Verification commands and outcomes

Focused suite:

```text
python -m pytest -q tests/test_chapter06_synthesis_oracle_worker.py
.........
9 passed in 3.28s
```

Targeted static checks:

```text
ruff check scripts/chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_oracle_worker.py
All checks passed!

ruff format --check scripts/chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_oracle_worker.py
2 files already formatted

python -m py_compile scripts/chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_oracle_worker.py
pass
```

An additional independent in-memory inventory check proved:

- exact frozen constants and accepted scope text;
- exact 754-member family decomposition;
- 8,451,402 object-free array bytes;
- exact catalog equality with the 13+5 mapping;
- exact eight-member factor-name set;
- full-factor shape, dtype, contiguity, finiteness, and bounds;
- center-sample equality for all eight factors;
- exact net reconstruction for every batched and loop route;
- exact `nonzero_count == 2 * reach + active`.

Separate path probes reproduced fail-closed behavior for both symlink aliases
and both ordinary alternate paths.

## 8. Final accepted boundary and remaining gates

The final accepted worker boundary is:

| property | accepted value |
| --- | --- |
| worker SHA-256 | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| test SHA-256 | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| key count | `754` |
| schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| full fingerprint | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` |
| capture scope complete | `True` |

The frozen-worker acceptance record may now bind this audit and these exact
identities.

CUDA/MPS measurement, compact-candidate construction, raw-to-final ownership,
deterministic serialization, detached authorization, publication, and
manifest synchronization remain later gates. This audit does not authorize
any of them.
