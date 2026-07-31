# Chapter 2 Exact Source Contract

## Authority, scope, and reader-facing rule

This audit is pinned to Payne Zero commit
`9c44001feae40b85146630499e6f8a5fed42e5af`. The source checkout at
`/Users/ysting/payne-zero` is a read-only development oracle.

Chapter 2 owns the numerical and data contracts that must be trustworthy before
the book fills an atmosphere with physical populations. It may expose exact
source names and exact source excerpts, but it must not invent a common wrapper
around the atmosphere and synthesis implementations. The two packages solve
the same mathematical integral with deliberately different layouts,
dependency graphs, and hardware policies.

This document covers:

- the readable NumPy parabolic depth integral and its private compiled
  equivalents;
- the real `prange` frequency-chunk accumulator;
- the Torch `(wavelength, depth)` optical-depth scan;
- the synthesis device and dtype policy;
- abundance notation and exact parser/representation boundaries;
- the complete native schema-v4 inventory and validator;
- the fixed-electron-density atmosphere-to-synthesis bridge;
- the adjacent data-identity functions Chapter 2 needs;
- the state of the progressively copied local source.

The later chemistry, opacity, transfer-physics, and atmosphere chapters own the
physical derivations inside these containers. Chapter 2 owns their names,
axes, units, storage identity, dependency order, and validation claims.

## Pinned source identities

| Module | SHA-256 |
| --- | --- |
| `payne_zero_atmosphere/radiative_transfer.py` | `df8970ca629487537a7c4849278eab5d755b527002d8fc58360c9264a3aa45db` |
| `payne_zero_atmosphere/transfer_kernels.py` | `50e759a085e6aefdb7819a3dbe3ef5e83405834f4b07e0a4de2f3c0e7354d3b9` |
| `payne_zero_synthesis/radiative_transfer.py` | `52e0d1a0c4a2713294ce1b43130c5d900e54c4cf1f8b2b05058fc2d6831ff62b` |
| `payne_zero_synthesis/device.py` | `22e769ebed60ad3a0f2060264247e469a99afd20ec5cadb69a01b6e5fa82ea3c` |
| `payne_zero_atmosphere/warm_start.py` | `3a83af3d68be52a35bfc3f55f5912770661be8251cc28f28da3250b2e83e0ad3` |
| `payne_zero_atmosphere/direct_abundance.py` | `ec65683eb344c4c3fd77340c084e780f58c6401e77c9f0d6db05ef6753131445` |
| `payne_zero_atmosphere/atmosphere_io.py` | `95c4d2cab230f6925e9404639ecb05b25af8c0c85755ac1ca70d760156a8683e` |
| `payne_zero_synthesis/atmosphere.py` | `06b79770e4d9472093655022d53ee7fddf7cc6727206f34c0f60c57151e2cf9b` |
| `payne_zero_synthesis/atmosphere_schema.json` | `2ba8d637e613be12ff43ce319a752616323f0341ea69f8e2391c3c244939777a` |
| `payne_zero_synthesis/equation_of_state.py` | `6497c29abb954e0b55d918cc22fa7b660952812c548faf1d7b1053345ef13562` |
| `payne_zero_synthesis/synthesis.py` | `590e430b6582fbcf601a52b721d8f65073432903773a99238073a1d821fe0d0c` |
| `payne_zero_synthesis/pipeline.py` | `465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22` |
| `payne_zero_synthesis/api.py` | `77718303c1e0052a520ece7fab277b3b1922c21d09b35a288596592d03310940` |
| `payne_zero_synthesis/cli.py` | `2cb2fa1cb71b5ec4a15dee9564b622c3e3288032ba13df59ae3355062b107dad` |
| `payne_zero_atmosphere/data_files.py` | `bf89c32977fc2db0454cf597718d99b3f3d15487529ecddbacf717ad6dc245c2` |
| `payne_zero_atmosphere/source_catalogs.py` | `a9ea21735c9d4964b785d76c89c9fc976a30ed75f8b6f9d4f7c6aaa4e77dae36` |
| `payne_zero_synthesis/paths.py` | `2bca3284eb1765449ab3fc87439eb603e3941213d9c1205c71aee3fd1ad30b5d` |

## 1. Readable NumPy depth integration

### Exact module and signatures

Module:
`payne_zero_atmosphere.radiative_transfer`

```python
def parabolic_coefficients(
    values: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

```python
def integrate_on_depth_grid(
    grid: np.ndarray,
    values: np.ndarray,
    *,
    surface_value: float,
) -> np.ndarray
```

`surface_value` is keyword-only in the public NumPy function. Do not change
that signature in teaching code.

### Shape, unit, dtype, and order contract

Let `D` be the number of depth points.

| Name | Shape | Unit | Dtype/device | Meaning |
| --- | --- | --- | --- | --- |
| `grid` / internal `coordinate` | `(D,)` | arbitrary coordinate unit \(x\); physically often g cm\(^{-2}\) | input is converted with `np.asarray(..., dtype=np.float64)` on CPU | monotonic depth coordinate |
| `values` / internal `value` | `(D,)` | \(q/x\) for an integral returning \(q\); physically often cm\(^2\) g\(^{-1}\) | converted to NumPy `float64` on CPU | integrand at each depth |
| `surface_value` | scalar | \(q\) | converted with `float(...)` by the public function | value of the cumulative integral at index 0 |
| `constant` | `(D,)` | same as `values` | NumPy `float64`, CPU | coefficient \(a\) in \(a+bx+cx^2\) |
| `linear` | `(D,)` | `values / grid` | NumPy `float64`, CPU | coefficient \(b\) |
| `quadratic` | `(D,)` | `values / grid**2` | NumPy `float64`, CPU | coefficient \(c\) |
| returned `integral` | `(D,)` | \(q\) | NumPy `float64`, CPU | ordered cumulative integral |

For the optical-depth use,

\[
\tau_\nu(m_i)
=
\tau_{\nu,0}
+
\int_{m_0}^{m_i}\chi_\nu(m')\,dm',
\]

`grid=column_mass` is in g cm\(^{-2}\), `values=extinction` is in
cm\(^2\) g\(^{-1}\), and the result and `surface_value` are dimensionless.

### Exact numerical semantics

1. `parabolic_coefficients` allocates three zeroed `float64` arrays.
2. Empty input returns three empty arrays.
3. A single value sets only `constant[0]`.
4. The first and last local fits are linear.
5. Interior fits are quadratic.
6. Intervals at zero-based indices 1 and, when present, 2 are deliberately
   forced linear near the surface.
7. Curved neighboring intervals are blended with an absolute-curvature
   weight. The weight is not a free chapter parameter.
8. The penultimate coefficient triplet is replaced by the final linear
   triplet.
9. `integrate_on_depth_grid` evaluates each interval analytically and advances
   `integral[index + 1]` from `integral[index]`. This recurrence is ordered in
   depth and is not a valid `prange` loop.

The exact functions do not explicitly validate equal lengths, one-dimensional
inputs, monotonicity, or distinct grid points. Zero denominators are locally
replaced by zero slopes. The chapter's declared input contract and tests must
therefore catch bad dimensionality, length, and depth ordering before the
kernel; it must not claim the function itself performs those validations.

### Bite-sized source treatment

`integrate_on_depth_grid` is short enough to show as one exact code card.
`parabolic_coefficients` is not. Keep its complete exact definition in the
progressive package, but display it in five consecutive exact excerpts:

1. conversion, allocation, and the `D=0/1` exits;
2. the two endpoint lines;
3. the interior quadratic fit;
4. the forced-linear near-surface intervals;
5. curvature blending and the final penultimate copy.

Each excerpt should be preceded by the numerical need it answers. Do not paste
the entire definition into one Markdown block.

## 2. Private compiled equivalents

### Exact module, decorator, and signatures

Module:
`payne_zero_atmosphere.transfer_kernels`

The module configures the Numba cache and treats Numba as a hard requirement.
The serial helpers are defined inside the module's Numba-available branch:

```python
_njit = numba.njit(cache=True, nogil=True)
```

```python
def _parabolic_coefficients_compiled(value, coordinate)
```

```python
def _integrate_on_depth_grid_compiled(coordinate, value, surface_value)
```

These leading-underscore functions are private production details. They are
not alternative public APIs.

### Exact semantics

- The private coefficient function has the same arithmetic and return order as
  the readable NumPy function.
- The private integrator has the same ordered interval recurrence.
- It allocates `float64` outputs and is a CPU function.
- `cache=True` permits compiled artifacts to be reused; it does not make an
  arbitrary first call free.
- `nogil=True` releases the Python interpreter lock while compiled code runs;
  it does not create threads.
- Neither private depth helper has `parallel=True`.
- Neither helper uses `prange`.

The correct Chapter 2 comparison is the exact NumPy function versus the exact
private compiled function on the same `float64` arrays. Do not manufacture a
parallel optical-depth helper.

### Bite-sized source treatment

Show `_njit = ...` and `_integrate_on_depth_grid_compiled` in full. For
`_parabolic_coefficients_compiled`, use a short AST/source-parity result beside
the same five conceptual movements already explained for the NumPy function.
Repeating both long coefficient bodies would add no new concept.

## 3. The real `prange` boundary

### Exact module and thread-count helper

Module:
`payne_zero_atmosphere.transfer_kernels`

```python
def transfer_chunk_count() -> int
```

The function returns `max(1, int(numba.get_num_threads()))`; its defensive
fallback is `max(1, os.cpu_count() or 1)`.

The exact caller in `payne_zero_atmosphere.runner.accumulate_transfer_state`
uses:

```python
chunk_count = min(
    transfer_chunk_count(),
    max(1, stop - start),
)
```

Therefore the normal caller never creates more chunks than active
frequencies.

### Exact parallel signature

```python
@numba.njit(parallel=True, nogil=True, cache=True)
def accumulate_transfer_range_parallel(
    chunk_count,
    range_start,
    range_stop,
    frequency_hz,
    frequency_weights,
    planck_all,
    stimulated_all,
    continuum_absorption_slab,
    continuum_scattering_slab,
    continuum_source_slab,
    line_mass_absorption_coefficient_slab,
    column_mass,
    h_over_kt,
    temperature,
    transfer_grid,
    mean_intensity_operator,
    eddington_flux_operator,
    second_moment_weights,
    target_integrated_eddington_flux,
    effective_temperature,
    frequency_count,
    rosseland_accumulator,
    radiation_energy_density,
    integrated_eddington_flux,
    radiative_acceleration,
    surface_radiation_pressure_constant,
    temperature_correction_heating_derivative,
    temperature_correction_mean_intensity_minus_source_integral,
    temperature_correction_integrated_eddington_flux,
    temperature_correction_diagonal_lambda,
)
```

There is no return value. The final nine array arguments are mutated in place.
The serial worker has the same signature without the leading `chunk_count`:

```python
def accumulate_transfer_range_compiled(
    range_start,
    range_stop,
    ...,
)
```

### Exact array layout and precision

Let `F` be the full opacity-sampling frequency count, `D` the depth count, and
`G=51` the fixed transfer-grid count in the pinned table.

| Argument group | Exact shape/layout | Unit or meaning | Caller dtype/device |
| --- | --- | --- | --- |
| `range_start`, `range_stop` | scalar integers, half-open `[start, stop)` | indices into the full frequency axis | CPU integers |
| `frequency_hz` | `(F,)` | Hz | contiguous NumPy `float64`, CPU |
| `frequency_weights` | `(F,)` | Hz quadrature width | contiguous NumPy `float64`, CPU |
| `planck_all` | `(F,D)` | \(B_\nu\), the module's per-frequency source unit | NumPy `float64`, CPU |
| `stimulated_all` | `(F,D)` | dimensionless \(1-e^{-h\nu/kT}\) | NumPy `float64`, CPU |
| `continuum_absorption_slab` | `(D,F)` | cm\(^2\) g\(^{-1}\) | contiguous NumPy `float64`, CPU |
| `continuum_scattering_slab` | `(D,F)` | cm\(^2\) g\(^{-1}\) | contiguous NumPy `float64`, CPU |
| `continuum_source_slab` | `(D,F)` | same spectral-radiance unit as \(B_\nu\) | contiguous NumPy `float64`, CPU |
| `line_mass_absorption_coefficient_slab` | `(D,F)` | cm\(^2\) g\(^{-1}\), before the local stimulated-emission multiplication | contiguous NumPy `float32`, CPU |
| `column_mass` | `(D,)` | g cm\(^{-2}\), increasing inward | contiguous NumPy `float64`, CPU |
| `h_over_kt` | `(D,)` | s | contiguous NumPy `float64`, CPU |
| `temperature` | `(D,)` | K | contiguous NumPy `float64`, CPU |
| `transfer_grid` | `(G,)` | dimensionless monochromatic optical depth | contiguous NumPy `float64`, CPU |
| `mean_intensity_operator` | `(G,G)` | fixed dimensionless quadrature/operator | loaded/cast to contiguous NumPy `float32`, CPU |
| `eddington_flux_operator` | `(G,G)` | fixed dimensionless quadrature/operator | loaded/cast to contiguous NumPy `float32`, CPU |
| `second_moment_weights` | `(G,)` | fixed dimensionless weights | loaded/cast to contiguous NumPy `float32`, CPU |
| `target_integrated_eddington_flux` | scalar | target bolometric \(H=\sigma T_{\rm eff}^4/(4\pi)\) | Python float |
| `effective_temperature` | scalar | K | Python float |
| `frequency_count` | scalar integer | full frequency count; activates the exact one-frequency controlled branch when 1 | CPU integer |
| each depth accumulator | `(D,)` | an internal frequency integral named by the argument | NumPy `float64`, CPU, mutated |
| `surface_radiation_pressure_constant` | `(1,)` | pre-finalization surface second-moment integral | NumPy `float64`, CPU, mutated |

The nine mutable targets are:

1. `rosseland_accumulator`;
2. `radiation_energy_density`;
3. `integrated_eddington_flux`;
4. `radiative_acceleration`;
5. `surface_radiation_pressure_constant`;
6. `temperature_correction_heating_derivative`;
7. `temperature_correction_mean_intensity_minus_source_integral`;
8. `temperature_correction_integrated_eddington_flux`;
9. `temperature_correction_diagonal_lambda`.

The names are exact. Their final physical normalization belongs to Chapters 12
and 13; Chapter 2 should identify them as internal pre-finalization
accumulators rather than prematurely assigning public output units.

### Exact independence and reduction semantics

1. `total_span = range_stop - range_start`.
2. Non-positive span or `chunk_count` returns without mutation.
3. Bounds are exact integer partitions:
   `range_start + (total_span * c) // chunk_count`.
4. Each chunk owns nine zero-initialized private buffers, all allocated as
   NumPy `float64`.
5. `numba.prange(chunk_count)` runs independent contiguous frequency ranges.
6. Each iteration calls the exact serial
   `accumulate_transfer_range_compiled` on its private buffers.
7. After the parallel region, a normal `range(chunk_count)` loop adds private
   buffers to the shared accumulators in increasing chunk order.

Frequency work is independent; depth recurrences inside one frequency and the
final chunk reduction remain ordered. A fixed chunk/thread policy is required
for the strongest reproducibility claim. Changing chunk count changes the
floating-point grouping and can change low bits.

### Precision wording audit

The source docstring calls the regrouped difference a “float32 reduction.”
The actual private and shared accumulator arrays are explicitly `float64`.
Float32 is nevertheless part of each frequency contribution because the line
slab, transfer operators, and transfer source grid are float32. The accurate
reader-facing statement is:

> Each frequency follows the same serial formal-solution path; chunking changes
> the grouping of the final floating-point accumulator additions. The
> per-frequency transfer path contains intentional float32 precision islands,
> while the private and shared frequency-integral buffers are float64.

Do not tell the reader that the chunk-private accumulator buffers themselves
are float32.

### Bite-sized source treatment

The complete parallel transfer call tree is too large for one Chapter 2 code
card. Keep the exact implementation hidden in the progressive package, then
show four exact excerpts:

1. decorator, signature, early return, and private-buffer allocation;
2. integer chunk-bound construction;
3. the `prange` call with one representative private-buffer lane;
4. the fixed-order reduction.

Use the complete positional call from `runner.accumulate_transfer_state` as a
machine-checked integration test, not a large Markdown listing. Chapter 12
owns the physical meaning of all nine accumulators.

## 4. Torch optical-depth integration

### Exact module and signatures

Module:
`payne_zero_synthesis.radiative_transfer`

```python
def _parabolic_interval_coefficients(
    values: torch.Tensor,
    depth_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

```python
def integrate_optical_depth(
    column_mass: torch.Tensor,
    extinction: torch.Tensor,
    surface_tau: torch.Tensor,
) -> torch.Tensor
```

The coefficient helper is private. `integrate_optical_depth` is the exact
module-level synthesis kernel.

### Shape, unit, dtype, device, and order contract

| Name | Shape | Unit | Dtype/device |
| --- | --- | --- | --- |
| `column_mass` | `(D,)` | g cm\(^{-2}\) | floating Torch tensor on the selected runtime device |
| `extinction` | `(W,D)` | cm\(^2\) g\(^{-1}\) | floating Torch tensor; its dtype/device determine the output |
| `surface_tau` | `(W,)` | dimensionless | compatible floating Torch tensor on the same device |
| coefficient arrays | `(W,D)` | extinction, extinction/column-mass, extinction/column-mass\(^2\) | `torch.zeros_like(extinction)` |
| `interval_width` | `(D-1,)` | g cm\(^{-2}\) | follows `column_mass` |
| `interval_tau` | `(W,D-1)` | dimensionless | broadcast Torch result |
| returned `optical_depth` | `(W,D)` | dimensionless | `torch.empty_like(extinction)` |

The exact production caller uses:

```python
surface_tau = extinction[:, 0] * column_mass[0]
optical_depth = integrate_optical_depth(column_mass, extinction, surface_tau)
```

Although the docstring calls `surface_tau` a top half-cell seed, the exact
caller shown above uses the product of top-layer extinction and top-layer
column mass. Preserve the executable expression.

Every wavelength row is independent. The only sequential dependency is the
prefix sum along depth:

```python
torch.cumsum(interval_tau, dim=1)
```

Do not transpose this into atmosphere-style `(depth, frequency)` in the public
contract. Do not claim `cumsum` makes the physical depth dependency
independent.

The helper mirrors the endpoint, near-surface linearization, curvature
blending, and penultimate-copy choices of the NumPy path, expressed through
Torch broadcasting. It assumes compatible shapes, devices, floating dtypes,
and a usable depth grid; it does not run the schema validator.

### Bite-sized source treatment

The public integrator is one appropriate code card. Split the longer private
coefficient helper into:

1. endpoint fits;
2. broadcast interior fit;
3. forced-linear top intervals;
4. detached curvature weight and final copy.

The `.detach()` on `blend_weight` is exact. Chapter 2 need only say that the
discrete interpolation weight is not differentiated through; it should not
turn this numerical chapter into a fitting chapter.

## 5. Exact synthesis device and dtype policy

Module:
`payne_zero_synthesis.device`

### Exact constants and functions

```python
ACCUMULATION_DTYPE = torch.float32
DEFAULT_DTYPE = torch.float32
REFERENCE_DTYPE = torch.float64
```

```python
def device() -> torch.device
```

```python
def resolve_runtime(
    requested_device: torch.device | str | None = None,
    requested_dtype: torch.dtype | None = None,
) -> tuple[torch.device, torch.dtype]
```

The import-time `DEVICE` is chosen by `_default_device()` in this exact order:

1. CUDA, when `torch.cuda.is_available()`;
2. MPS, when `torch.backends.mps.is_available()`;
3. CPU.

`device()` returns that stored `DEVICE`; it does not rescan hardware on every
call.

`resolve_runtime` applies:

| Runtime device | Omitted dtype | Explicit `torch.float64` |
| --- | --- | --- |
| CUDA | `REFERENCE_DTYPE == torch.float64` | allowed |
| CPU | `REFERENCE_DTYPE == torch.float64` | allowed |
| MPS | `DEFAULT_DTYPE == torch.float32` | raises `ValueError` |

`ACCUMULATION_DTYPE` is always `torch.float32`, independent of work dtype.
Later line deposition must retain that separate policy.

The module also contains:

```python
def to_dev(
    value,
    dtype=DEFAULT_DTYPE,
    device: torch.device | None = None,
) -> torch.Tensor
```

Its omitted `dtype` is literally `DEFAULT_DTYPE`, hence float32 even on
CUDA/CPU. The public runtime policy comes from `resolve_runtime`, not from
assuming that an unqualified `to_dev(value)` produces CUDA/CPU float64.

Chapter 2 must distinguish:

- array location (`device`);
- arithmetic representation (`dtype`);
- the intentional line-accumulation precision island;
- allocation/transfer time;
- kernel time with device synchronization;
- the final host crossing.

The physical atmosphere iteration remains NumPy/Numba CPU code.

## 6. Abundance notation and exact parser boundaries

### Physical notation introduced before APIs

Use:

\[
A(X)=\log_{10}(n_X/n_H)+12,
\]

\[
[X/H]
=
\log_{10}(n_X/n_H)_\star
-
\log_{10}(n_X/n_H)_\mathrm{ref},
\]

\[
[X/Fe]=[X/H]-[Fe/H].
\]

`dex` denotes a base-10 logarithmic interval. `[M/H]` is a global
metal-pattern coordinate, not one element. `[alpha/M]` modifies the exact
seven-element set below. These bracket quantities must never be passed where
the structured schema expects linear number fractions.

### Standard-pattern source boundary

Module:
`payne_zero_atmosphere.warm_start`

Exact constants:

```python
HELIUM_NUMBER_FRACTION = 0.078370
ALPHA_ELEMENT_ATOMIC_NUMBERS = (8, 10, 12, 14, 16, 20, 22)
```

The tuple corresponds to O, Ne, Mg, Si, S, Ca, and Ti.

`SOLAR_METAL_LOG_ABUNDANCES_3_TO_99` is a `float64` `(97,)` table indexed by
`atomic_number - 3`. Its values are base-10 logarithms of number fractions in
the mixture convention used by the fixed-column deck; it is not a
schema-ready linear `(99,)` vector and not an `A(X)` array.

Exact functions:

```python
def compute_metal_log_number_abundances(
    metallicity: float = 0.0,
    alpha_enhancement: float = 0.0,
    absolute_abundance_offsets: Mapping[int, float] | None = None,
) -> np.ndarray
```

```python
def compute_hydrogen_fraction(
    metallicity: float = 0.0,
    alpha_enhancement: float = 0.0,
    absolute_abundance_offsets: Mapping[int, float] | None = None,
) -> float
```

Exact order:

1. copy the solar `(97,)` metal table;
2. add `metallicity` to every metal;
3. add `alpha_enhancement` to the exact seven alpha elements;
4. replace any valid element in `absolute_abundance_offsets` with
   `solar[element] + offset`;
5. compute H as
   `1 - HELIUM_NUMBER_FRACTION - sum(10**metal_log_number_abundances)`.

The individual offset therefore overrides both scalar-pattern shifts for that
element. The compute function silently ignores offset keys outside `3..99`;
the parser below is the boundary that rejects them. `compute_hydrogen_fraction`
does not independently assert that its result is positive.

### Text parser

Exact signature:

```python
def parse_abundance_offset(text: str) -> tuple[int, float]
```

Accepted examples:

- `"Fe:+0.3"` → `(26, 0.3)`;
- `"26:+0.3"` → `(26, 0.3)`;
- symbols are case-insensitive and surrounding fields are stripped.

Exact rejection boundary:

- the text must contain exactly one `:`;
- a numeric element field must satisfy `.isdigit()`;
- a nonnumeric field must be a known element symbol;
- the atomic number must lie in `3..99`.

The value is converted with `float(offset_text)`. This parser does not itself
reject `nan` or `inf`; later public direct-abundance normalization does. Do not
claim the parser is a full physical abundance validator.

The synthesis CLI's private `_parse_abundances(entries, abundance_file)`:

- accepts repeated `--abundance` strings and, optionally, one JSON object;
- converts JSON entries to the same `key:value` text form;
- calls `parse_abundance_offset`;
- rejects duplicate atomic numbers;
- returns `dict[int, float]`.

It is a CLI parser, not a new abundance representation.

### Direct-abundance boundary

Module:
`payne_zero_atmosphere.direct_abundance`

Exact constants relevant here:

```python
DIRECT_XH_FAMILY = "direct_abundance"
DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX = 0.01
```

`DIRECT_XH_ATOMIC_NUMBERS` contains the 81 elements in `Z=3..99` with a finite
public solar reference under the pinned threshold. The other 16 solver slots
are `DIRECT_XH_SENTINEL_ATOMIC_NUMBERS`.

Exact signatures:

```python
def complete_direct_abundance_vector(
    abundance_by_atomic_number: Mapping[int, float],
) -> np.ndarray
```

```python
def retained_direct_abundance_mixture(
    *,
    iron_abundance_relative_to_hydrogen: float,
    retained_abundance_relative_to_iron_by_atomic_number: Mapping[int, float],
) -> tuple[dict[int, float], np.ndarray]
```

`complete_direct_abundance_vector` requires all 81 public `[X/H]` values,
fills each of the 16 nonpublic solver slots from `[Fe/H]`, quantizes all 97
values to 0.01 dex, normalizes signed zero, applies the exact support checks,
and returns NumPy `float64[97]`.

`retained_direct_abundance_mixture` takes one `[Fe/H]` and selected `[X/Fe]`
coordinates. Every unselected public element inherits `[Fe/H]`; selected
elements use `[X/H]=[Fe/H]+[X/Fe]`. It returns the complete 81-entry public
mapping and the realized quantized `(97,)` solver vector. Iron must not appear
in the retained map.

At the higher public label API,
`payne_zero_synthesis.api._normalize_sparse_direct_xh` accepts symbol or atomic
number keys, rejects booleans, nonintegral numeric keys, duplicates, and
nonfinite values. Sparse unspecified public elements inherit `fe_over_h`
before the complete 81-entry mapping is passed to
`complete_direct_abundance_vector`. `metallicity` and `alpha_enhancement` do
not apply in direct-abundance mode.

Chapter 2 should establish these representation boundaries, not teach the
experimental initializer network; Chapter 14 owns that model and its mandatory
closure gate.

### Fixed-column deck to linear schema boundary

Module:
`payne_zero_atmosphere.atmosphere_io`

```python
def linear_elemental_abundances(model: ModelAtmosphere) -> np.ndarray
```

The function returns NumPy `float64[99]`, indexed by `Z-1`:

- stored H and He (`Z <= 2`) are already linear and are copied;
- stored metals (`Z >= 3`) are decoded with `10**stored_value`;
- keys outside `1..99` are ignored;
- any missing element in `1..99` raises `ValueError`;
- the result is not divided by its sum and is not silently renormalized.

This is the exact bridge from fixed-column deck storage to
`elemental_abundances`. It must not be replaced by a book-defined
`abundance_to_linear` API.

### Abundance representation fence

| Representation | Exact shape/index | Unit/meaning | May substitute for another row? |
| --- | --- | --- | --- |
| `SOLAR_METAL_LOG_ABUNDANCES_3_TO_99` and standard-pattern result | `(97,)`, `Z-3` | log10 number fraction for metals | no |
| public direct labels | mapping over 81 supported `Z` | `[X/H]` dex | no |
| direct solver vector | `(97,)`, `Z-3` | quantized `[X/H]` dex | no |
| fixed-column deck abundance block | mapping `Z -> stored value` | H/He linear, metals logarithmic | no |
| schema `elemental_abundances` | at least `(99,)`, `Z-1` | positive linear relative number abundance | no |

### Bite-sized source treatment

The two standard-pattern functions and `parse_abundance_offset` are each small
enough for one code card. Show `linear_elemental_abundances` in one card. Split
`complete_direct_abundance_vector` into:

1. exact key/completeness checks;
2. 97-slot fill and 0.01-dex quantization;
3. support checks.

Do not show the full direct-initializer module in Chapter 2.

## 7. Native schema-v4 validator

### Exact module, constants, and functions

Module:
`payne_zero_synthesis.atmosphere`

```python
ATMOSPHERE_SCHEMA_VERSION = 4
POPULATION_ION_STAGE_COUNT = 6
POPULATION_SPECIES_COUNT = 139
ATMOSPHERE_PRODUCT_METADATA_SCHEMA_VERSION = 1
LEGACY_ATMOSPHERE_SCHEMA_VERSIONS = (1, 2, 3)
```

Exact reader functions:

```python
def load_atmosphere_npz(path: str | Path) -> dict[str, np.ndarray]
```

```python
def load_atmosphere_product_metadata(
    path: str | Path,
) -> dict[str, object] | None
```

```python
def validate_atmosphere_npz(path: str | Path) -> tuple[str, ...]
```

`validate_atmosphere_npz` accepts only `path`, calls the exact loader, and
returns `REQUIRED_ATMOSPHERE_ARRAYS`. Do not add validation flags to its
signature.

### Complete required array inventory

The declarative source is
`payne_zero_synthesis/atmosphere_schema.json`. There are 25 required public
array names:

| Exact field | Shape | Unit/representation |
| --- | --- | --- |
| `temperature` | `(D,)`, `D >= 2` | K |
| `gas_pressure` | `(D,)` | dyn cm\(^{-2}\) |
| `electron_density` | `(D,)` | cm\(^{-3}\) |
| `mass_density` | `(D,)` | g cm\(^{-3}\) |
| `column_mass` | `(D,)` | g cm\(^{-2}\), strictly increasing |
| `partition_normalized_populations` | `(D,6,139)` | cm\(^{-3}\) per partition function |
| `ion_stage_populations` | `(D,6,139)` | actual ion-stage cm\(^{-3}\) |
| `fractional_doppler_widths` | `(D,6,139)` | dimensionless \(v/c\) |
| `hydrogen_neutral_population` | `(D,)` | cm\(^{-3}\) |
| `helium_neutral_population` | `(D,)` | cm\(^{-3}\) |
| `helium_singly_ionized_population` | `(D,)` | cm\(^{-3}\) |
| `molecular_hydrogen_population` | `(D,)` | cm\(^{-3}\) |
| `hydrogen_partition_normalized_ion_stage_populations` | `(D,2)` | cm\(^{-3}\) per partition function |
| `carbon_partition_normalized_ion_stage_populations` | `(D,2)` | cm\(^{-3}\) per partition function |
| `magnesium_neutral_partition_normalized_population` | `(D,)` | cm\(^{-3}\) per partition function |
| `aluminum_neutral_partition_normalized_population` | `(D,)` | cm\(^{-3}\) per partition function |
| `silicon_neutral_partition_normalized_population` | `(D,)` | cm\(^{-3}\) per partition function |
| `iron_neutral_partition_normalized_population` | `(D,)` | cm\(^{-3}\) per partition function |
| `hc_over_kt` | `(D,)` | cm |
| `microturbulence` | `(D,)` | cm s\(^{-1}\) |
| `elemental_abundances` | `(n_element,)`, `n_element >= 99` in validator | positive linear relative number abundance |
| `signed_continuum_edge_frequency_hz` | `(E,)`, `E >= 2` | signed Hz; every entry nonzero |
| `continuum_edge_wavelength_nm` | `(E,)` | positive nm |
| `continuum_edge_midpoint_wavelength_nm` | `(E-1,)` | positive nm |
| `continuum_edge_interval_width_squared_over_two_nm2` | `(E-1,)` | positive nm\(^2\) |

The signed frequency field carries an interpolation-side sign convention.
Only its magnitude is a physical frequency.

The public `build_structured_atmosphere` documentation asks for exactly
`elemental_abundances.shape == (99,)`; the storage validator deliberately
accepts any one-dimensional array with at least 99 positive entries. State
which boundary is being discussed.

### Exact validation behavior

For a native schema-v4 product, the loader:

1. opens with `np.load(..., allow_pickle=False)`;
2. reads an optional scalar integer `atmosphere_schema_version`;
3. rejects unsupported versions;
4. rejects legacy aliases when the declared version is 4;
5. requires all 25 canonical arrays;
6. returns arrays under canonical public names;
7. checks every loaded array is numeric and finite;
8. checks the common depth axis;
9. requires positive `temperature`, `gas_pressure`, `electron_density`,
   `mass_density`, `column_mass`, and `hc_over_kt`;
10. requires nonnegative `microturbulence`, populations, and Doppler widths;
11. requires strictly increasing `column_mass`;
12. requires all three population/Doppler cubes to share `(D,6,139)`;
13. requires the H and C two-stage arrays to have `(D,2)`;
14. requires a positive one-dimensional abundance vector of length at least
    99;
15. checks compatible edge and interval lengths, nonzero signed frequencies,
    and positive wavelength-side edge arrays.

The loader contains read-only compatibility for versions 1–3 and can
reconstruct actual ion-stage populations for pre-v3 products. Chapter 2 should
mention this as an existing read boundary but write and teach only canonical
schema v4.

The schema does not prescribe one storage dtype. The validator accepts any
numeric finite dtype; it does not cast loaded arrays to float64. Production
builders normally emit physical arrays as NumPy float64 and
`atmosphere_schema_version` as a one-element NumPy int32 array. Do not claim
“schema v4 means float64 storage” as a validator guarantee.

### Same shape does not mean same physics

For actual ion population \(n\) and partition function \(U\),

\[
\texttt{ion_stage_populations}=n,
\qquad
\texttt{partition_normalized_populations}=n/U.
\]

Both arrays have `(D,6,139)`. They are not interchangeable. The actual cube is
needed by charge-weighted free-free opacity; the partition-normalized cube is
the starting quantity for LTE bound-level populations. The validator checks
names, shape, finiteness, and sign, but does not prove the identity
`actual == normalized * partition_function`.

### Product-metadata extension

The all-or-none exact field tuple is:

```python
ATMOSPHERE_PRODUCT_METADATA_FIELDS = (
    "atmosphere_product_metadata_schema",
    "atmosphere_product_role",
    "atmosphere_converged",
    "atmosphere_closure_required",
    "initializer_family",
    "atmosphere_metadata_json",
)
```

`load_atmosphere_product_metadata`:

- returns `None` when none of the six fields is present;
- rejects an incomplete extension;
- requires schema 1;
- requires scalar integer schema, scalar string role/family/JSON, and scalar
  Boolean safety fields;
- parses one JSON object;
- requires typed role, family, convergence, and closure fields to agree with
  their JSON copies;
- returns a dictionary containing exact keys `schema`,
  `atmosphere_product_role`, `atmosphere_converged`,
  `atmosphere_closure_required`, `initializer_family`, `labels`,
  `provenance`, and `timings`.

This metadata loader is separate from `validate_atmosphere_npz`. Array-schema
validity does not establish convergence or physical closure.

### What validation does not prove

The exact validator does not prove:

- normalization of `elemental_abundances`;
- charge conservation;
- consistency between actual and partition-normalized populations;
- radiative equilibrium;
- hydrostatic closure;
- atmosphere iteration convergence;
- correct source-data identity;
- numerical parity with the pinned implementation;
- spectral accuracy.

Chapter 2 must keep the four claims separate:

1. checksum identity;
2. schema validity;
3. numerical parity;
4. physical acceptance.

### Bite-sized source treatment

Keep the full exact loader and validator in the progressive module. Display:

1. constants and the required-name tuple as a rendered table, not a huge code
   block;
2. schema-version and canonical-name checks;
3. depth/positivity/order checks;
4. population-representation shape checks;
5. edge-grid checks;
6. the three-line `validate_atmosphere_npz` public boundary;
7. metadata all-or-none and typed/JSON agreement as separate short excerpts.

## 8. Fixed-electron-density bridge

### Reader-facing promise

There are two physically different operations:

- a full charge solve determines `electron_density`;
- the atmosphere-to-synthesis bridge preserves an already supplied
  `electron_density` while filling the synthesis population fields.

Chapter 2 establishes that interface promise. Chapter 3 derives the atomic
population and charge equations, and Chapter 4 adds molecular equilibrium.

### Exact public entry

Module:
`payne_zero_synthesis.api`

```python
def build_structured_atmosphere(
    *,
    temperature,
    column_mass,
    gas_pressure,
    electron_density,
    elemental_abundances,
    mean_nuclear_mass_amu: float | None = None,
    microturbulence=None,
    mass_density=None,
    molecular_lines: bool = True,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    eos_tolerance: float = 1.0e-5,
) -> dict[str, np.ndarray]
```

The documented input units are:

| Input | Shape | Unit/representation |
| --- | --- | --- |
| `temperature` | `(D,)` | K |
| `column_mass` | `(D,)` | g cm\(^{-2}\), outer-to-inner and increasing |
| `gas_pressure` | `(D,)` | dyn cm\(^{-2}\) |
| `electron_density` | `(D,)` | cm\(^{-3}\), already solved |
| `elemental_abundances` | `(99,)` | linear number fractions at indices `Z-1` |
| `microturbulence` | scalar or `(D,)` | cm s\(^{-1}\) |
| `mass_density` | optional `(D,)` | g cm\(^{-3}\) |
| `mean_nuclear_mass_amu` | optional scalar | atomic mass units |

### Exact call chain

```text
payne_zero_synthesis.api.build_structured_atmosphere
  → payne_zero_synthesis.synthesis.build_structured_atmosphere_from_columns
    → resolve_runtime(device, dtype)
    → EOSTables.from_npz(device=runtime_device, dtype=runtime_dtype)
    → payne_zero_synthesis.pipeline.build_structured_atmosphere_from_columns
      → equation_of_state.solve_population_state_at_electron_density
      → pack canonical schema-v4 mapping
```

The engine explicitly passes:

```python
electron_density_seed=electron_density
```

The pipeline then passes that exact array to
`solve_population_state_at_electron_density`; it does not call
`solve_population_state`, the full charge-balance path.

### Exact fixed-density signature

Module:
`payne_zero_synthesis.equation_of_state`

```python
def solve_population_state_at_electron_density(
    temperature,
    gas_pressure,
    elemental_abundances,
    *,
    tables: EOSTables,
    electron_density,
    mean_nuclear_mass_amu=None,
    mass_density=None,
    molecules: bool = False,
    molecules_path=None,
) -> PopulationState
```

Exact semantics:

1. convert `temperature`, `gas_pressure`, and `electron_density` to NumPy
   float64;
2. evaluate the EOS tables at the supplied electron density;
3. compute
   `total_nuclei_number_density = max(P/(kT) - electron_density, floor)`;
4. use supplied `mass_density` when present, otherwise derive it from mixture;
5. construct `ElectronDensityResult` with the supplied electron-density array;
6. assemble actual and partition-normalized population fields without a
   charge-balance fixed-point loop.

The Torch EOS tables use the runtime device/dtype selected by
`resolve_runtime`. The population bundle returned across this bridge consists
of host NumPy float64 physical arrays plus its internal Torch `eos` record.
On MPS, table work uses float32; on CUDA/CPU the omitted runtime dtype is
float64. The supplied electron-density column itself remains the host
float64 array.

### Exact `PopulationState` fields

```python
@dataclass
class PopulationState:
    electron_density: np.ndarray
    total_nuclei_number_density: np.ndarray
    mass_density: np.ndarray
    partition_normalized_populations: np.ndarray
    ion_stage_populations: np.ndarray
    hydrogen_neutral_population: np.ndarray
    hydrogen_ionized_population: np.ndarray
    hydrogen_partition_normalized_ion_stage_populations: np.ndarray
    helium_neutral_population: np.ndarray
    helium_singly_ionized_population: np.ndarray
    carbon_partition_normalized_ion_stage_populations: np.ndarray
    magnesium_neutral_partition_normalized_population: np.ndarray
    aluminum_neutral_partition_normalized_population: np.ndarray
    silicon_neutral_partition_normalized_population: np.ndarray
    iron_neutral_partition_normalized_population: np.ndarray
    eos: EOSResult
    molecular_populations: Optional[np.ndarray] = None
    molecular_equation_densities: Optional[np.ndarray] = None
```

The important output shapes are:

| Field group | Shape | Unit |
| --- | --- | --- |
| density columns | `(D,)` | cm\(^{-3}\), except `mass_density` in g cm\(^{-3}\) |
| `partition_normalized_populations` | `(D,6,139)` | cm\(^{-3}\) per partition function |
| `ion_stage_populations` | `(D,6,139)` | cm\(^{-3}\) |
| H/C per-ion partition-normalized arrays | `(D,2)` | cm\(^{-3}\) per partition function |
| species-specific neutral/ion columns | `(D,)` | actual or partition-normalized as spelled in the exact field name |

When molecules are enabled, molecular equilibrium can change molecular
population fields and, if no mass density was supplied, the density scale.
It still does not replace the supplied electron-density column with a new
charge solution.

### Bridge boundary checks

The Chapter 2 fixture should verify:

- returned `electron_density` equals the supplied array under the declared
  representation policy;
- the output uses all canonical schema-v4 names;
- actual and partition-normalized cubes are both present and distinct;
- depth order and units remain unchanged;
- the fixed-density and full-charge functions are called in separate tests;
- no claim of charge closure is made for a deliberately arbitrary fixed
  electron-density fixture.

Do not create a `FixedElectronBridge` class or renamed wrapper. Use the exact
public builder and exact EOS function names.

### Bite-sized source treatment

Show:

1. the public builder signature and unit docstring;
2. the engine line that forwards `electron_density_seed=electron_density`;
3. the fixed-density function through construction of `ElectronDensityResult`;
4. the `PopulationState` field list;
5. a compact call-trace test proving the full charge solver is not invoked.

The population assembly body belongs to Chapters 3 and 4 and should not be
dumped into Chapter 2.

## 9. Adjacent data-identity boundary

Chapter 2's checksum lesson should use the exact source functions rather than
a book-defined manifest API.

### Atmosphere source catalogs

Module:
`payne_zero_atmosphere.source_catalogs`

```python
def source_catalog_root() -> Path
def load_source_catalog_checksums(
    checksum_path: Path | None = None,
) -> dict[str, str]
def verify_source_catalog_checksums(
    *,
    root: Path | None = None,
    checksum_path: Path | None = None,
) -> dict[str, object]
def source_line_paths(root: Path | None = None) -> dict[str, Path]
def atmosphere_source_catalog_paths(root: Path | None = None) -> dict[str, Path]
```

`source_catalog_root` resolves the exact environment variable
`PAYNE_ZERO_SOURCE_CATALOG_ROOT`, then the bundled tree.

`load_source_catalog_checksums` rejects malformed hashes, absolute/parent
paths, duplicate entries, and empty manifests. The committed identity file is
`source_data_files/source_catalogs/CHECKSUMS.sha256`.

`verify_source_catalog_checksums` returns exact keys:

- `status`;
- `root`;
- `checksum_manifest`;
- `checksum_manifest_sha256`;
- `file_count`;
- `total_bytes`;
- `files`.

Each file record has `path`, `bytes`, and `sha256`.

The `source_line_paths` docstring says the returned set includes H3+ lines,
but the executable dictionary has no `h3plus_lines_path` entry. H3+ remains an
optional `AtmosphereInput` field and is consumed only when an explicit existing
path is supplied. The chapter must present the dictionary it actually returns;
it must not promise that the standard resolver supplies H3+.

### Data-root functions

Use exact names:

- `payne_zero_atmosphere.data_files.data_root`;
- `atmosphere_table_dir`;
- `atmosphere_table_path`;
- `atmosphere_emulator_dir`;
- `payne_zero_synthesis.paths.data_root`;
- `source_catalog_root`;
- `source_catalog_path`.

The runtime data manifest and the source-catalog checksum file have different
roles. A checksum establishes byte identity, not physical correctness.

## 10. Local progressive-source audit

Audit snapshot: 2026-07-30, while Chapter 2 integration is in progress.

The current executable gate

```text
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_pinned_source_fragments.py
```

passes against the pinned commit.

### Present and exact

- `src/payne_zero_synthesis/constants.py` is byte-identical.
- `src/payne_zero_synthesis/device.py` is byte-identical.
- `src/payne_zero_atmosphere/_numba_cache.py` is byte-identical.
- `src/payne_zero_atmosphere/transfer_kernels.py` is now the complete
  byte-identical pinned module, including both serial and parallel
  accumulators.
- `src/payne_zero_synthesis/atmosphere.py` is now the complete byte-identical
  pinned schema loader/validator module.
- `src/payne_zero_atmosphere/data_files.py` and
  `src/payne_zero_atmosphere/source_catalogs.py` are byte-identical.
- `src/payne_zero_synthesis/radiative_transfer.py` contains exact AST matches
  for `planck_bnu`, `_parabolic_interval_coefficients`, and
  `integrate_optical_depth`.
- `src/payne_zero_atmosphere/radiative_transfer.py` contains exact AST matches
  for `parabolic_coefficients` and `integrate_on_depth_grid`.

There is no detected mismatch in any definition or file currently declared
exact by `src/PAYNE_ZERO_SOURCE_MANIFEST.json`.

### Coverage gaps that must not be mistaken for mismatches

The progressive package does not yet contain these Chapter 2 source
boundaries:

- `atmosphere_schema.json`;
- the standard-pattern abundance functions and parser;
- the fixed-column abundance decoder;
- the direct-abundance completion functions;
- the public structured-atmosphere builder and its fixed-\(n_e\) call chain;
- the synthesis-side path resolver used by the complete package.

This is acceptable only while they are described as audited future additions.
It becomes a reader-facing mismatch if Chapter 2 claims they are executable
from the local progressive package. Before that claim, either:

1. copy the exact complete dependency-bearing modules and extend the manifest
   and verifier; or
2. clearly label a supplied result as a source-audit fixture and defer
   execution to the owning later chapter.

For the parallel accumulator, the self-contained route is to keep the complete
exact transfer-kernel call tree in the progressive source while displaying
only the four bite-sized excerpts listed above. Copying only the outer
function without its private serial dependencies would not be executable.

### Three source/prose discrepancies to handle explicitly

1. The parallel-kernel docstring describes a “float32 reduction,” but the
   chunk-private and shared accumulator buffers are allocated as float64.
   Preserve the exact source and use the more precise reader-facing wording in
   Section 3.
2. The Torch optical-depth docstring says “top half-cell seed,” while the exact
   production caller sets
   `surface_tau = extinction[:, 0] * column_mass[0]`. Preserve and test the
   executable expression.
3. `source_line_paths` describes H3+ as part of its returned “full” set, but
   the exact return dictionary omits `h3plus_lines_path`; that field is an
   optional explicit configuration path. Do not add an invented resolved H3+
   member to the dictionary.

Neither discrepancy authorizes a cleaned-up replacement API.

## 11. Chapter 2 source acceptance checklist

Chapter 2 is source-faithful only when all of the following are true:

- every displayed production definition is AST-equal to the pinned source;
- long exact definitions live in source and appear only as causally separated
  excerpts in the narrative;
- `integrate_on_depth_grid` remains `(D,)` NumPy float64 CPU with a
  keyword-only `surface_value`;
- `_integrate_on_depth_grid_compiled` remains serial;
- the only `prange` demonstration is the real frequency-chunk accumulator;
- its private buffers and fixed chunk-order reduction are visible;
- atmosphere slabs remain `(depth, frequency)`;
- synthesis optical depth remains `(wavelength, depth)` with
  `cumsum(dim=1)`;
- CUDA/CPU default to float64 work, MPS to float32, and accumulation stays
  float32 where the exact synthesis policy says so;
- abundance dex, deck storage, and linear schema arrays are never conflated;
- all 25 canonical schema-v4 fields are named exactly;
- schema validity is not presented as physical closure;
- the fixed-\(n_e\) bridge preserves supplied electron density and is not
  described as a charge solve;
- checksum identity, schema validity, numerical parity, and physical
  acceptance remain four distinct claims;
- no wrapper API, renamed field, universal axis order, or universal constant
  namespace is introduced.
