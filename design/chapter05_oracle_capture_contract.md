# Chapter 5 continuum oracle capture contract

Status: implementation-ready, unpublished comparison capture  
Worker: `scripts/chapter05_oracle_worker.py`  
Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

## 1. Purpose and publication boundary

The worker executes the exact pinned atmosphere and synthesis continuum
implementations and returns deterministic NumPy arrays in memory. Its sole
external physical input is
`data/fixtures/chapter05_continuum_states.npz`, SHA-256
`ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae`.

This is not a golden publisher. It reads no golden artifact, has no output-file
option, and records `meta__golden_publication_performed = False`. A later
reviewed publisher may serialize a selected subset only after scientific and
governance review.

The fixture is accepted only at its canonical local path. Its complete
217-member key set, regime order, dtypes, finite values, and positive physical
fields are validated before any continuum kernel runs.

The physical payload also contains a digest over all 217 raw fixture members,
including each sorted field name, dtype, shape, and C-order byte sequence.
Consequently the physical fingerprint binds the exact state arrays themselves,
not only the external fixture filename or file hash. The fixture's atmosphere
and synthesis states are accepted Chapter 4 outputs; Chapter 5 does not
recompute or silently regularize them.

## 2. Fail-closed identity and process contract

The worker requires:

- the exact read-only root `/Users/ysting/payne-zero`;
- the pinned Git commit above;
- a frozen 52-file manifest covering the exact set of pinned Payne Zero Python
  modules loaded by the two continuum engines, the atmosphere runner, and the
  synthesis pipeline;
- byte equality for every file in that manifest before import and exact-set
  equality for every dynamically loaded pinned `.py` file after import;
- the exact `runner.py` binding to
  `compute_continuum_opacity_columns` and the exact `pipeline.py` binding to
  the audited synthesis continuum module;
- byte hashes for the atmosphere continuum, Karzas-Latter, and molecular
  tables and for the synthesis continuum and edge-grid tables;
- `PAYNE_ZERO_DATA_ROOT` equal to the pinned source-data directory;
- all imported `payne_zero_atmosphere` and `payne_zero_synthesis` modules
  resident under the pinned root;
- an external `NUMBA_CACHE_DIR` that exists as a real directory and contains
  no entry at process start;
- one Numba, Torch, BLAS, OpenMP, and expression-evaluator thread;
- CPU Torch tensors with explicit `torch.float64`.

`PYTHONHASHSEED=0`, `LC_ALL=C`, `TZ=UTC`, disabled user site packages, and
disabled bytecode writes are required before the process begins. A mismatched
source, table, fixture, environment, module path, shape, or physical-domain
condition aborts the capture.

The unresolved cache path is inspected with `lstat` and `is_symlink` before
resolution. A symlink is rejected. The exact 52-file loaded-source set and
hashes are checked once after the imports and again after every capture lane
has executed.

The capture metadata includes SHA-256 digests of the worker itself and this
contract, plus frozen capture-schema version `2`. These identities enter the
full-capture fingerprint.

## 3. Four declared stellar regimes

Every main product is captured independently for the fixture's frozen order:

1. `hot_dwarf`;
2. `solar_dwarf`;
3. `low_gravity_giant`;
4. `cool_molecule_rich`.

Each regime has six atmosphere layers. The worker never recomputes the
equation of state; the accepted Chapter 4 atmosphere and synthesis states in
the fixture are the declared Chapter 5 inputs.

## 4. Atmosphere product capture

The exact entry is
`payne_zero_atmosphere.continuum_opacity.compute_continuum_opacity_columns`.
The call always receives the fixture's explicit 20-element runner flag vector;
the permissive `opacity_flags=None` default is never used.

For every regime the actual atmosphere product is evaluated exactly as
`runner.prepare_opacity_state` places it: the effective-temperature-selected
30,000-point wavelength grid is converted to frequency and passed to the
flagged continuum composite. The captured product arrays are therefore:

- `product_absorption (6,30000)`;
- `product_scattering (6,30000)`;
- `product_source (6,30000)`.

The runner source hash and its live binding to the continuum function are part
of the fail-closed identity contract. The Rosseland branch is inactive in the
frozen runner flag vector, so no surrogate table enters these products.

A separate compact diagnostic frequency vector contains nextafter triplets
below, at, and above nine source thresholds:

- H-minus bound-free;
- the 20,000 cm\(^{-1}\) H2-CIA cutoff;
- the CH lower and upper photon-energy bounds;
- the OH lower and upper photon-energy bounds;
- the H2-plus upper frequency;
- the H I Rayleigh cap;
- the H2 Rayleigh cap.

On that compact diagnostic the capture records:

- absorption, scattering, and absorption-weighted source;
- named pre-total absorption and source arrays for H I, H-minus, H2-plus,
  neutral He, ionized He, He-minus, the molecular composite, neutral C/Mg/Al/
  Si/Fe, lukewarm metals, and hot metals;
- separate electron, H I Rayleigh, He I Rayleigh, and H2 Rayleigh scattering;
- ordered component sums and residuals against the public total;
- CH, OH, and H2-CIA arrays before their molecular sum.

No nonfinite-value repair or positivity clamp is applied by the worker.
Finite, nonnegative full products and compact diagnostic opacities are
validation conditions. The independently reconstructed absorption-weighted
source must also close within the common `2e-12` scale-relative gate.

## 5. Atmosphere sampling and line-reference capture

`build_opacity_sampling_grid` is captured at every threshold temperature in
the fixture:

`4499, 4500, 7249, 7250, 12999, 13000, 29999, 30000 K`.

Both full `(8, 30000)` wavelength grids and frequency quadrature weights are
retained. Each regime also records its full effective-temperature-selected
30,000-point grid.

The exact 344-entry continuum reference wavelength and packed-index arrays are
captured. For each regime the worker:

1. obtains the active reference indices and frequencies;
2. evaluates the exact flagged continuum at those frequencies;
3. calls `assemble_continuum_line_selection_threshold`;
4. retains active absorption, scattering, source, and the final `(6, 344)`
   float32 threshold.

This preserves the sentinel column and the stimulated-emission division used
for line selection without treating the threshold as raw continuum opacity.

## 6. Atmosphere molecular boundary capture

The fixture's H2 policy temperatures
`100, 101, 8999, 9000, 19899, 19900, 20000, nextafter(20000,+inf) K`
drive exact H2 equilibrium-constant and population calls with declared,
constant normalized-H and departure inputs. The capture retains input bit
patterns and the effective 100–19,900 K table temperature.

CH and OH cross-section-times-partition grids are evaluated at 8999 and
9000 K on nextafter probes at both molecular photon-energy bounds. This pins
the per-layer 9000 K gate separately from the column-global molecular-entry
gate exercised by the four physical regimes.

H2-CIA is evaluated at 3000, 3500, and 4000 K and at 5000 cm\(^{-1}\) plus a
nextafter triplet around 20,000 cm\(^{-1}\). These small synthetic probe arrays
are constants declared inside the worker, not an external input artifact.
Their population, helium, density, and stimulated-emission inputs are all
retained.

The source's reversed-looking temperature convention is recorded explicitly:

```text
table column (index - 1) weight = temperature_fraction
table column index       weight = 1 - temperature_fraction
```

The worker stores the indices, fractions, both weights, and the resulting H2-H2
and H2-He logarithmic coefficients. It does not reinterpret or “correct” this
parity-pinned order.

## 7. IFOP 4/13 coupling

The cool molecule-rich atmosphere is evaluated three ways:

- IFOP 4 only;
- IFOP 13 only;
- IFOP 4 and 13 together.

The exact source constructs the H I ground population only in the IFOP 4
branch, so IFOP 13 alone must produce zero scattering. The capture retains all
three slabs and the H2 Rayleigh increment
`both - IFOP4`. A nonzero IFOP-13-only result aborts the worker.

IFOP 19 is captured separately with a populated Rosseland-style table and an
otherwise zero flag vector. The surrogate must produce nonzero, frequency-flat
absorption by depth, zero scattering, and its bolometric-like frequency-flat
source. It is never blended into the ordinary continuum product.

## 8. Molecular-entry and H2 ownership counterfactuals

The atmosphere molecular composite is additionally executed on:

- an all-warm six-layer column with `min(T)=9000 K`, which must remain exactly
  zero;
- an otherwise identical mixed column with one 8999 K layer, which must enter
  the molecular composite;
- the cool physical state with `ch_population=oh_population=0`, which must
  retain exactly the continuum-local H2-CIA term while CH and OH remain zero.

A three-layer owner probe at
`19999, 20000, nextafter(20000,+inf) K` separately captures H2 CIA and H2
Rayleigh. Both owners must be active in the first two layers and exactly
inactive in the third. This distinguishes the H2 partition-table clamp from
the later, strict `T > 20000 K` opacity cutoff.

## 9. Standard synthesis product

The exact product entry is
`payne_zero_synthesis.continuum.continuum`. The worker passes the same trimmed
18-field mapping constructed by `SynthesisPipeline`, deliberately omitting the
optional actual-H-II and stored-H2 fields. `build_pops` is therefore entered by
the product call, and the H-II normalized-stage fallback is exercised.

The edge-rich requested wavelength vector uses interior-left, midpoint, and
interior-right samples in eleven fixed intervals from the extreme UV through
the mid-infrared, plus the fixture's nextafter triplet around edge 170. The
worker records:

- requested wavelengths and their one-sided `searchsorted(..., side="right")`
  interval indices;
- unique used intervals;
- all edge-triplet frequencies and the selected sample indices/frequencies;
- signed edge frequencies and all edge geometry;
- left, center, and right parabolic basis values and their sum;
- final interpolated absorption and scattering;
- absorption and scattering at the selected pre-interpolation samples;
- an independent NumPy reconstruction from the floored sample logarithms and
  the three stored basis functions;
- reconstruction residuals against the exact Torch product.

The standard route is explicitly marked
`coulomb_table_energy_first=False` and
`frequency_invariants_supplied=False`.

The complete 1,020-member edge-sample vector rebuilt by
`build_edge_sample_frequencies` must be bit-identical to the packaged vector.
A temporary trace around `_compute_at_freqs` proves that the standard call
evaluates exactly the three samples for each used interval and no sample from
an unused interval.

Three additional standard-route counterfactuals are executable:

- the trimmed pipeline H II fallback is compared with a rich mapping carrying
  actual `hydrogen_ionized_population`; a 1.5-times rich-H-II perturbation must
  alter absorption but never scattering;
- changing the stored schema `molecular_hydrogen_population` by
  `17 * value + 1` must leave absorption and scattering bit-identical;
- reversing every signed edge frequency must leave both products
  bit-identical because the sampler consumes edge magnitudes.

## 10. Synthesis component capture

During an independent exact `_compute_at_freqs` call on the standard route's
selected sample frequencies, temporary wrappers observe and return unchanged:

- H-minus absorption;
- H I absorption;
- minor-term absorption and scattering;
- neutral- and ionized-helium absorption;
- hot-metal absorption;
- the light-element/Si II return;
- H I Rayleigh and electron scattering.

The wrappers are restored in a context manager even if the call fails.
Component columns, ordered sums, and residuals against `_compute_at_freqs` are
retained. Components are intentionally captured before logarithmic parabolic
interpolation: interpolated opacities are not linear sums of independently
interpolated component logs. The independent final interpolation
reconstruction is nevertheless executable and must pass the same `2e-12`
scale-relative gate.

The standard scalar `_minor_terms` return is not left as an opaque remainder.
Seven isolated population-state calls capture:

- H2-plus absorption;
- He-minus absorption together with the independently named He I Rayleigh
  scattering returned by that state;
- compact C I, Mg I, Al I, and Si I absorption separately;
- H2 Rayleigh scattering.

Their ordered absorption and scattering sums must reproduce the unmodified
`_minor_terms` columns element by element.

## 11. Sampled diagnostic capture

For every regime,
`compute_sampled_continuum(..., frequency_invariants=None)` is evaluated on the
same threshold-rich frequency vector as the compact atmosphere diagnostic. It
records absorption, scattering, Planck source, and the same observed scalar
component columns.

This lane is explicitly marked
`coulomb_table_energy_first=True` and
`frequency_invariants_supplied=False`. It is a sampled diagnostic, not the
standard `SynthesisPipeline.run` product and not the precomputed-invariants
extension.

## 12. Finite sampled precomputed-invariants extension

The implemented extension is captured explicitly rather than being implied by
the scalar diagnostic. Its declared support is the twelve-point wavelength
vector

`100, 125, 160, 200, 250, 320, 400, 500, 700, 1000, 1600, 2500 nm`.

For each regime the worker:

1. uses the same trimmed pipeline atmosphere and exact `build_pops` result;
2. calls `build_frequency_invariants(...,
   coulomb_table_energy_first=True)`;
3. calls `compute_sampled_continuum` with that exact invariants object;
4. requires finite, nonnegative `(6,12)` absorption and scattering and a
   finite `(6,12)` source.

The capture retains the wavelength and frequency inputs, support bounds,
layout flag, `frequency_invariants_supplied=True`, the exact 18 input
atmosphere field names, all 25 numeric `pops` arrays, every numeric
`FrequencyInvariants` field, their ordered field-name inventories, and the
three output slabs. The mutable derived tensor cache is intentionally excluded.
This is an implemented extension with a tested finite support window, not the
standard synthesis product and not a claim of global scalar equivalence.

## 13. Grid counts, regime activation, and exact schema

The eight atmosphere grid-boundary probes must produce active line-reference
counts

`226, 240, 240, 263, 263, 299, 299, 338`.

The repeated 263 and 299 counts are explicit regression assertions, not merely
printed metadata. Every named atmosphere and standard-synthesis component in
the four physical regimes must activate somewhere on its compact diagnostic
grid, and every layer of each full atmosphere product must contain nonzero
absorption and scattering.

Principal contracts are checked directly, including:

- full atmosphere products `(6,30000)` float64;
- compact atmosphere products `(6,27)` float64;
- line thresholds `(6,344)` float32;
- boundary grids `(8,30000)` float64;
- standard synthesis products `(6,36)` float64;
- sampled diagnostics `(6,27)` float64;
- sampled extensions `(6,12)` float64.

Every remaining product, component, counterfactual, trace, input, and
intermediate is fail-closed through a sorted key/dtype/shape schema digest.
Accepted schema version `2` has exactly `1161` keys and digest
`652c110dc79a6f6dfca6893bee35416289675b4920a5d0dcfe6b2cb262dacf3d`.
The completion bit cannot become true unless both accepted constants match.
No object-dtype field is permitted.

## 14. Determinism and acceptance checks

Two fingerprints have distinct meanings:

- `meta__physical_payload_fingerprint` hashes every physical input, grid,
  boundary, component, intermediate, and output outside the `meta__` and
  `identity__` namespaces;
- `meta__full_capture_fingerprint` also hashes all environment, version,
  source, table, fixture, scope, and contract metadata.

Only the two self-referential fingerprint fields are excluded from either
digest. Main products must be finite; absorption and scattering must be
nonnegative. Residuals are checked element by element, never against one
global maximum. Each comparison uses relative tolerance `2e-12` and an
explicit absolute floor: `1e-20` for opacity/component sums, `1e-30` for
source reconstruction, and `1e-20` for interpolation.

The atmosphere source reconstruction begins from an independently evaluated
Planck array for zero-absorption cells; it is never seeded from the public
source output. The IFOP coupling assertion, source reconstruction, minor-term
split, synthesis interpolation, and parabolic basis partition-of-unity
assertions must all hold.

The command-line interface prints only a JSON summary. It exposes no
serialization or publication option.

Create a new empty cache directory for each canonical invocation:

```bash
NUMBA_CACHE_DIR="$(mktemp -d /tmp/chapter05-oracle.XXXXXX)"
env \
  LC_ALL=C TZ=UTC PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 MKL_DYNAMIC=FALSE MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" \
  python scripts/chapter05_oracle_worker.py
```

Reusing that directory must fail before Payne Zero is imported.

## 15. Regression acceptance protocol

Accepted key-count and schema-digest constants were added only after two
independent empty-cache candidate captures agreed exactly. Because the full
fingerprint binds this contract's own SHA-256, final fingerprints are reported
in the task handoff after the frozen contract is written; embedding that
fingerprint back into this file would create a self-referential identity loop.

The final acceptance requires two additional independent empty-cache captures
with `meta__capture_scope_complete=True`, followed by a negative invocation
against a populated cache. No array archive or golden may be written.
