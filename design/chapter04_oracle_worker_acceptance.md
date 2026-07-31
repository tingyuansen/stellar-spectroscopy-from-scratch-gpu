# Chapter 4 Oracle Worker Acceptance Re-audit

Status: **PASS — oracle workers accepted for the publisher stage**  
Audit date: 2026-07-30  
Pinned implementation:
`9c44001feae40b85146630499e6f8a5fed42e5af`

This is the second independent audit of:

- `scripts/chapter04_atmosphere_oracle_worker.py`;
- `scripts/chapter04_synthesis_oracle_worker.py`;
- their focused test modules;
- `design/chapter04_oracle_capture_contract.md`; and
- `design/chapter04_exact_source_contract.md`.

The re-audit specifically retested every blocker in the first report, the
atmosphere zero-code priming/cache schedule, the independent CO
line-population reconstruction, and the fixed-route supplied-mass input
branch. No worker, test, source module, manifest, data file, external checkout,
or golden product was changed.

## 1. Executable evidence

The complete focused suite now passes:

```text
22 passed in 23.35s
```

Ruff also passes on both workers and tests:

```text
All checks passed!
```

The atmosphere test itself executes two fresh processes with independent
Numba cache directories and requires identical fingerprints. The synthesis
test executes every scientific route in a fresh pinned process. This re-audit
ran all five synthesis routes once more under the complete frozen environment,
so every synthesis digest below was independently reproduced.

| In-memory result | Keys | Reproduced digest |
| --- | ---: | --- |
| atmosphere complete implemented lifecycle | 460 | `a6116c5f73c7ed3b0ee51907c419a307de2e1477be09f0b9969fab888c0b7682` |
| synthesis full | 196 | `f4e36d39a9b736ade95972d810801e7ecaad66a5c7c0985b83ca897f0485b8d0` |
| synthesis fixed, derived mass | 155 | `ce0cb3c2a2d323011dfe4b7a0a4ea1dc9a37df2ad5dbad22bd6482539995a56f` |
| synthesis fixed, supplied mass | 156 | `1e9782da2c9990bbefa74fd34d26dd72d8a8445773e8b646d773a7a9935a4444` |
| synthesis public structured mapping | 244 | `2e87bbc3b8bf09b03af232515518513daa0d1886e120b677355496c333bfee96` |
| synthesis boundaries | 112 | `49b633ef299deb2ad3d37009506a79daa1931c941997e742b7fdd7fd4b8c62f1` |

These are stable, sorted in-memory array fingerprints, not published NPZ
hashes. Archive-byte reproducibility remains a later publisher gate.

## 2. Former blocker closure

### Atmosphere Newton history — closed

Every atmosphere Newton trace now has lossless layer offsets, layer indices,
iteration indices, pre-update equation densities, raw corrections, maximum
relative updates, `still_iterating` values, and exclusive direct/fallback
linear-solver masks.

For the main six-depth route:

- iterations are `[9,5,5,5,7,7]`;
- the concatenated history has 38 rows;
- pre-update densities and raw corrections each have shape `(38,23)`;
- `np.linalg.solve` was attempted 38 times;
- the direct branch completed 38 times;
- `np.linalg.lstsq` was used zero times; and
- the final history value at every depth reconstructs the stored convergence
  and stopping ratio.

The same history self-check runs for the handoff, modes 2 and 12, energy mode,
temperature controls, pressure controls, and boundary probes. Both NumPy
bindings are restored after capture.

### Exact boundary bit patterns and masks — closed

The atmosphere result stores exact `uint64` values, branch masks, and result
bit patterns.

- Atmosphere 10000 K: `4666723172467343360`,
  next-after `4666723172467343361`.
- Atmosphere 20000 K: `4671226772094713856`,
  next-after `4671226772094713857`.
- Polynomial active mask: `[true,false,false,false]`.
- Catalog H2 active mask: `[true,true,true,false]`.
- H2 helper temperatures include exact bits for
  100, 101, 19899, 19900, 20000, and the next value above 20000 K.
- H2 interpolation mask:
  `[false,true,true,false,false,false]`, with separate low/high-clamp and
  catalog-activity masks.

Synthesis likewise stores and checks:

- 10000 K bits
  `[4666723172467343360,4666723172467343361]`;
- polynomial mask `[true,false]`;
- 9000 K bits
  `[4666173416653455360,4666173416653455361]`; and
- provisional-H2 mask `[true,false]`, which agrees with the positive/zero
  population mask.

The above-gate synthesis polynomial constants are exactly zero and their
finite log sentinels are exactly `-700`.

### Complete catalog constants and semantic alignment — closed

The synthesis worker now serializes every raw array from both fixed-buffer
catalogs, plus:

- active molecule, equation, and component masks;
- decoded component species codes;
- explicit inverse-electron sentinel masks;
- the hard-coded molecular mass authority and its raw-byte SHA-256;
- code-aligned row indices;
- lossless shared-component offsets and semantic streams; and
- coefficient, component, and combined mismatch masks.

The executable result proves:

- atmosphere: 170 molecules, 23 equations, 481 components;
- synthesis: 190 molecules, 23 equations, 548 components;
- exactly 170 code-aligned shared records;
- exactly zero atmosphere-only records;
- exactly 20 synthesis-only records;
- 64 shared records move row position, proving that comparison is not by row;
  and
- zero coefficient or component-semantic mismatches among shared records.

The 20 synthesis-only codes are exactly:

```text
111, 10811, 10812, 10820, 60606, 60608, 60614, 60816, 61414,
61616, 70708, 70808, 80814, 80816, 1010106, 1010107, 1010606,
6060707, 101010106, 101010114
```

### Source and numerical-library provenance — scientifically closed

The atmosphere result records 35 executed pinned source modules. The
synthesis routes record 17. Both retain module names, pinned-relative paths,
and source SHA-256 values. Both now record platform, byte order, Python,
NumPy, backend versions, BLAS/LAPACK identity, thread counts, and declared
process controls.

The final one-variable process-control mismatch is closed in section 6. All
executed-source, library-identity, and process-environment omissions from the
first audit are now closed.

### Fresh-process synthesis regression tests — closed

The focused synthesis test now launches real pinned subprocesses for:

1. the two-solve full route;
2. fixed density with derived mass;
3. fixed density with supplied mass;
4. the public structured mapping; and
5. exact boundary probes.

Each route checks its digest, key count, call count, principal shape,
iterations, exhaustion count, catalog alignment, source count, platform,
byte order, BLAS identity, and recorded environment count. The independent
re-audit reproduced every expected result.

## 3. Atmosphere priming and cached schedule ownership

The full route captures the source's two distinct population phases rather
than attributing the solve to a later packed job.

1. One runner-level `populate_species` call uses code `0`, mode `1`, output
   shape `(6,1)`, and temperature iteration index `1`.
2. That zero-code call changes the cache from absent (`-1`) to `1` and owns
   exactly one molecular solve.
3. The following one `populate_all_species` call sees cache value `1`.
4. Its 230 packed jobs all see cache value `1`, solve count `1`, and
   temperature iteration index `1` before and after the job.
5. Therefore the packed schedule performs exactly zero additional molecular
   solves.

The schedule inventory contains 198 atomic jobs and 32 molecular jobs. The
molecular jobs are 16 exact codes, each appearing as the verified redundant
mode pair `[1,11]` in the partition-normalized target:

```text
101, 106, 107, 108, 112, 114, 120, 124,
126, 606, 607, 608, 814, 822, 823, 10108
```

Their one-based packed starts are:

```text
841, 846, 847, 848, 851, 853, 858, 862,
864, 868, 869, 870, 889, 895, 896, 940
```

The runner, EOS schedule, molecular, and NumPy wrappers are all restored.
This closes the priming/cache ownership seam without altering the source
arithmetic.

## 4. Independent public-lane and CO reconstruction

The public route still makes exactly:

- one fixed EOS call;
- one molecular solve;
- one no-ground partition call with `nion=6`;
- one production molecular-line mapping call;
- one edge-grid call; and
- zero fallback solves.

The diagnostic reconstruction does not call the production line mapper. It
independently transforms the equation densities, uses the no-ground neutral
partitions, exact catalog components, `1.8786e20`, `T/11604.5`, the reference
Saha factor, and the hard-coded parity-rounded atomic masses. It reproduces
all 54 public lanes bitwise.

For CO specifically:

- line species code is 276;
- equilibrium code is 608;
- components are atomic numbers `[6,8]`;
- molecular mass is `28.009999999999998` amu;
- leading coefficient is `11.091`;
- destination is `[depth,5,45]`;
- independent and public values are bitwise equal at all six depths; and
- raw equilibrium CO differs from normalized CO at all six depths, proving
  that the fixture distinguishes those observables.

The independently grounded counterfactual differs from the no-ground result.
Only the intended stage-5 molecular cells change; the actual ion cube is
bitwise unchanged. The exact continuum-edge arrays and solved catalog H2
replacement also pass.

## 5. Supplied-mass input-only branch

The supplied mass is calculated before the EOS call from only:

- fixture temperature;
- fixture gas pressure;
- fixture elemental abundances; and
- the pinned `atomic_masses.npz` array.

It uses

\[
\rho_{\rm supplied}
=\frac{P}{k_{\rm B,ref}T}
\frac{\sum_Z A_Zm_Z}{\sum_Z A_Z}
(1.660\times10^{-24}\ {\rm g\,amu^{-1}}).
\]

The exact six-depth input is:

```text
[4.585251757562519e-10, 3.981929157883241e-09,
 3.362517955545847e-08, 2.751151054537511e-07,
 2.1616186857080447e-06, 1.6812589777729235e-05]
```

Changing the fixture electron seed by a factor of seven leaves this mass
bitwise unchanged. The route makes one molecular solve, has iterations
`[8,5,5,5,7,7]`, does not exhaust, preserves public electron density, and
returns the supplied mass bitwise unchanged. It neither opens nor derives
from an oracle output.

## 6. Final process-control closure

The synthesis worker now includes `"NUMBA_NUM_THREADS": "1"` in
`THREAD_ENVIRONMENT`, carries it into `ORACLE_PROCESS_ENVIRONMENT`, requires
it before a scientific result is built, and serializes it with the other
controls.

This re-audit opened the in-memory result of every real synthesis route and
verified all three facts independently:

| Route | Serialized key present | Serialized value | Environment count |
| --- | --- | --- | ---: |
| full | yes | `"1"` | 12 |
| fixed, derived mass | yes | `"1"` | 12 |
| fixed, supplied mass | yes | `"1"` | 12 |
| public structured mapping | yes | `"1"` | 12 |
| boundaries | yes | `"1"` | 12 |

The exact key is `meta__environment__NUMBA_NUM_THREADS`. Every route's new key
count and digest matches the regenerated pinned values in section 1. The
focused regression test also asserts a 12-control environment and the exact
`NUMBA_NUM_THREADS` declaration. The former P1 reproducibility blocker is
therefore closed.

## 7. Deferred markers and output discipline

The atmosphere worker now marks its in-memory lifecycle scope complete while
keeping only `golden_publication` deferred. `require_complete=True` still
fails before solving, so the worker cannot be mistaken for a publisher.

Neither worker reads or writes a golden. No output-fed fixture or supplied
input was found. Results are copied, object-free, flat, lexically sorted
array mappings. Shared calculations reload only the immutable fixture and
byte-identified static assets.

The structural Doppler exception remains exact: 12 positive infinities occur
only at packed slots 919 and 927 over six depths; their isotope masses and
actual/normalized populations are zero, and every other Doppler value is
finite.

## 8. Gate decision

**Oracle-worker acceptance gate: PASS.** All former P0 and P1 blockers,
priming/cache ownership, both synthesis solve-ownership routes, independent
54-lane/CO reconstruction, exact boundaries, catalog alignment, process
provenance, structural Doppler behavior, and output-input isolation pass.

**Central publisher entry gate: PASS.** The accepted workers may now be used
to build the five candidate Chapter 4 archives. This does not itself publish
or accept those archives: the publisher must still produce two byte-identical
deterministic NPZ builds in fresh processes, verify archive schemas and
manifests, and keep candidate computation complete before any golden is
opened.
