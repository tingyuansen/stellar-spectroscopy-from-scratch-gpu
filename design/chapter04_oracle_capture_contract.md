# Chapter 4 Pinned-Oracle Capture Contract

Status: central implementation contract; golden publication is not yet
accepted  
Pinned source: Payne Zero commit
`9c44001feae40b85146630499e6f8a5fed42e5af`

This contract translates the accepted Chapter 4 source audit into an
executable comparison design. It governs the atmosphere and synthesis oracle
workers, the publisher, the manifest entries, and the later parity tests.
It does not govern the reader's calculation: reader-facing code must compute
its state before opening any golden archive.

## 1. Why the oracle is split

Chapter 4 crosses two independent production implementations:

- the atmosphere path is ordered NumPy/Numba/LAPACK `float64` on one CPU
  thread;
- the synthesis path is an ordered Torch molecular solve followed by batched
  post-solve evaluation, run here on CPU with an explicit `float64` dtype.

One archive cannot honestly describe both numerical lifecycles. Five
comparison products therefore have single responsibilities:

1. `chapter04_molecular_constants_cpu_float64.npz`
2. `chapter04_atmosphere_molecular_state_cpu_float64.npz`
3. `chapter04_synthesis_molecular_full_cpu_float64.npz`
4. `chapter04_synthesis_molecular_fixed_cpu_float64.npz`
5. `chapter04_molecular_public_mapping_cpu_float64.npz`

The constants archive owns catalog structure, formation-policy boundary
probes, and exact identities. The atmosphere archive owns the full route,
structured handoff, disabled route, lifecycle modes, continuation, and
molecular internal energy. The two synthesis state archives keep the
two-solve full route distinct from the one-solve fixed route. The public
mapping archive owns the structured builder, no-ground partition convention,
molecular public lanes, H2 replacement, and edge-grid dependency.

No archive is an input to another oracle calculation. Shared inputs are
reloaded from the immutable Chapter 4 fixture and static data.

## 2. Isolation and identity

Every worker starts in a fresh subprocess. Before importing NumPy, Numba, or
Torch, the publisher fixes:

```text
PYTHONHASHSEED=0
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
NUMBA_NUM_THREADS=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
MKL_DYNAMIC=FALSE
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
LC_ALL=C
TZ=UTC
```

The worker places `/Users/ysting/payne-zero` first on `sys.path`, asserts the
exact pinned commit, and rejects every imported `payne_zero_atmosphere` or
`payne_zero_synthesis` module whose resolved path is outside that checkout.
It records:

- Python, NumPy, Numba, and Torch versions where applicable;
- platform, byte order, device, dtype, thread controls, and available BLAS
  identity;
- the fixture hash;
- every executed source-file hash;
- every static-table or catalog hash used by the route.

The synthesis worker sets `torch.set_num_threads(1)` and
`torch.set_num_interop_threads(1)`. Every tensor and every public call receives
`device="cpu"` and `dtype=torch.float64` explicitly. It does not change the
global default dtype.

Archives are sorted, pickle-free deterministic NPZ files written through
`scripts/deterministic_npz.py`. Canonical JSON is stored as UTF-8 `uint8`
bytes or scalar Unicode arrays, never object arrays.

## 3. Shared input-only fixture

All state routes start from
`data/fixtures/chapter04_molecular_inputs.npz`. It supplies the six-depth
temperature, gas-pressure, column-mass, microturbulence, abundance, and
declared electron-seed track. It also supplies exact
`nextafter` boundary values.

The fixture contains no solved molecular population, converged electron
density, mass density, or golden output. An oracle output may never be copied
back into it.

Atmosphere `ModelAtmosphere` construction follows the accepted Chapter 3
contract:

- H and He deck values remain linear;
- metal deck values are `log10` of their positive linear abundances;
- `rosseland_opacity` is one only to satisfy the structural input contract;
- unused scaffold columns are zero;
- declared microturbulence and electron seed are passed unchanged;
- the atmosphere molecular catalog path is explicit.

Synthesis loads a fresh `EOSTables` instance from the pinned partition bundle
for each independent route. The synthesis molecular catalog, atomic-mass
table, and continuum-edge grid are explicit inputs whenever their public
interface permits one.

## 4. Non-mutating observation rule

The oracle may observe hidden production state only by temporarily wrapping
the exact binding that the route calls. A wrapper must:

1. preserve every original positional argument and keyword;
2. call the saved original exactly once;
3. copy diagnostics only after or immediately around that call;
4. return the same object or original public tuple;
5. restore the binding in `finally`.

It may not patch arithmetic, convergence, seeds, constants, dtype, device, or
operation order. Inspection-only wrappers and scoped line tracing are
allowed; replacement kernels and reimplemented solvers are not.

## 5. Atmosphere capture

The primary archive runs the real six-depth
`prepare_population_state(..., temperature_iteration_index=1)` route.
It captures:

- declared inputs and returned electron, nuclei, mass, charge-square, packed,
  Doppler-support, and energy arrays;
- actual layer seeds and pressure-scaled continuation;
- frozen Newton constants and post-solve population constants;
- every pre-update density and raw correction needed to reconstruct the final
  stopping ratio;
- iteration counts, final `still_iterating`, and `solve` versus `lstsq`
  counts;
- physical equation densities before transformation;
- saved physical equation rows;
- transformed equation densities;
- raw and partition-normalized molecular populations;
- exact post-solve residuals and row-scaled residuals.

The archive also uses fresh, independent states for:

- molecule-enabled structured handoff;
- pressure-iteration-disabled behavior;
- one-layer temperature controls at 3500, 6000, and 9000 K;
- one-layer pressure controls at 3500 K and
  \(10^2,10^4,10^6\ {\rm dyn\,cm^{-2}}\);
- exact 10000 K and 20000 K boundary/`nextafter` lifecycle probes;
- modes 2 and 12;
- molecular specific-internal-energy mode with saved physical rows.

Independent one-layer controls are never stacked into one depth track, because
stacking would introduce continuation into an intended control.

Atmosphere exhaustion is

```text
iteration_count == 200 and final_still_iterating
```

and is a hard capture failure. Molecular energy mode must leave normalized
arrays and saved physical rows byte-identical while using the nonzero saved
row as the actual Newton seed.

## 6. Synthesis full and fixed captures

The synthesis worker wraps the molecular solve binding used by the atomic EOS.
The wrapper requests the existing diagnostics tuple from the saved original,
captures it, and returns the original public four-tuple to an unchanged
caller.

The exact full call must produce two molecular solves in order:

1. the electron-closure call owned by `solve_electron_density`;
2. the final molecule-backed state call.

The published electron density must equal call 0's electron result. The
published molecular populations, molecular equation densities, and nuclei
density must equal call 1's results. Residual diagnostics remain route-local:
call 1 is never evaluated with call 0's electron array merely because that
array is public.

The exact fixed route is captured twice from fresh state:

- derived mass, with `mass_density=None`;
- a deterministic supplied mass computed only from the fixture and pinned
  atomic-mass table.

Each fixed call must make exactly one molecular solve. The public electron
density remains bitwise equal to the supplied input, while the internal
molecular equation's electron density is stored separately. The supplied-mass
branch must return its input mass bitwise unchanged.

For every molecular call, store:

- actual temperature, pressure, electron seed, abundance, ion constants,
  catalog, device, dtype, tolerance, maximum iterations, and chain length;
- all four returned arrays;
- active counts, codes, component structure, formation constants, and
  iterations completed;
- final raw and row-scaled residuals;
- pre-replacement molecular log-term finiteness diagnostics;
- active and padded-lane invariants.

Because `iterations_completed == max_iter` is ambiguous, a scoped trace
records execution of the exact source's exhaustion assignment. Any true
exhaustion flag is a hard capture failure.

## 7. Public structured mapping

The public archive calls the real
`build_structured_atmosphere_from_columns` once with molecular lines enabled,
the fixture's fixed electron density, explicit `eos_tables` resident on CPU in
`float64`, the exact `molecular_species_codes`, and the exact
`molecules_path`. The builder has no direct `device`, `dtype`, or
`molecular_lines` keyword; those policies are carried by the tables and by a
nonempty molecular-species-code array.

Observation wrappers establish:

- one fixed-density EOS call;
- one molecular solve with no fallback re-solve;
- one no-ground partition-function call with `nion=6`;
- the exact 54 requested molecular species codes and returned populations;
- one exact edge-grid construction.

The archive stores:

- every public structured numeric array, schema key, shape, and dtype;
- the pre-map fixed population state;
- partition and ion-stage cubes before and after molecular mapping;
- a lossless offset representation from the 54 public species codes to all
  contributing molecule codes;
- public column indices;
- line populations computed with grounded and no-ground partitions;
- the named H2 replacement and exact edge arrays.

The public lane must equal the no-ground result exactly and must be measurably
different from the grounded control. Ion-stage populations must remain
unchanged. Only the intended molecular cells in the normalized partition cube
may change.

## 8. Boundary probes

The constants archive distinguishes policies rather than presenting one
fictional universal molecular constant.

Atmosphere probes include:

- H2 helper interpolation at 100, 101, 19899, 19900, 20000 K and the next
  representable value above 20000 K;
- catalog-gated H2 constants from the real layer constant builder;
- ordinary polynomial constants at 10000 K and its next representable value.

Synthesis probes include:

- polynomial constants at 10000 K and its next representable value;
- the exact post-preprocessing finite log sentinel;
- the provisional public H2 9000 K gate before solved code 101 replaces it.

Each `nextafter` input is stored by exact bit pattern together with the branch
mask. A malformed polynomial-overflow case belongs in an isolated failure
test, not a production golden.

## 9. Comparison policy

Capture-time checks are exact. Commit, hashes, strings, key sets, source
locations, counts, call order, branch masks, padding, structural zeros, copied
inputs, fixed public electron density, supplied mass, saved energy rows, and
edge arrays must be bitwise equal to their declared authority.

For the same pinned source, environment, dtype, and operation order, local
parity first attempts exact array equality. If a computed nonzero field differs
because of measured library-level floating-point behavior, the parity test may
use a field-specific tolerance only after recording:

- maximum absolute error;
- scale-relative error;
- resolved-value-relative error;
- exact-zero leakage;
- log-space error for positive multi-decade quantities.

One broad tolerance for all densities or populations is forbidden. CPU
`float32`, CUDA, MPS, and alternate-thread profiles are separate live parity
experiments, not looser comparisons to the CPU `float64` archive.

## 10. Publication gate

No Chapter 4 golden enters `data/golden/payne_zero` until:

1. both focused workers pass structural self-tests;
2. the central publisher verifies the pinned commit without changing it;
3. two fresh captures produce byte-identical deterministic archives;
4. every expected call count and lifecycle invariant passes;
5. no layer exhausts;
6. every non-padding canonical result is finite, apart from the exact
   atmosphere structural exception
   `fractional_doppler_widths[:, [919, 927]] == +inf`; these two unused
   zero-mass packed slots must have exactly zero actual and
   partition-normalized populations, and every other Doppler slot must be
   finite;
7. archive schemas, shapes, units, and provenance are reviewed;
8. manifest entries bind each archive to its worker, source, and fixture;
9. reader code and teaching tests pass with the golden directory made
   unavailable;
10. comparison tests open goldens only after the corresponding local
    computation completes.

Only then may the Chapter 4 narrative state a pinned parity result.
