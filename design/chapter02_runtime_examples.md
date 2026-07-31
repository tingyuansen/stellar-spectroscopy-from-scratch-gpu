# Chapter 2 runtime examples

## Scope and authority

These examples were executed against the progressive local copies of the
Payne Zero modules pinned to commit
`9c44001feae40b85146630499e6f8a5fed42e5af`. They are integration notes for
the canonical Chapter 2 author, not additional chapter sections. The existing
schema fixture,
`data/fixtures/chapter02_schema_v4_minimal.npz`, is deliberately a synthetic
four-depth interface fixture; none of the values below should be presented as
a converged stellar atmosphere.

## 1. Standard mixture: exact names and mixed representations

The smallest example that makes both generic scaling and an element override
visible is:

```python
from payne_zero_atmosphere.warm_start import (
    compute_hydrogen_fraction,
    compute_metal_log_number_abundances,
)

mixture = dict(
    metallicity=-0.50,
    alpha_enhancement=+0.20,
    absolute_abundance_offsets={8: +0.10, 26: -0.30},
)
metal_log_number_abundances = compute_metal_log_number_abundances(**mixture)
hydrogen_fraction = compute_hydrogen_fraction(**mixture)
```

Exact selected results:

| Quantity | Result | Representation |
| --- | ---: | --- |
| H | `0.920874144997871` | linear number fraction |
| O, `metal_log_number_abundances[8 - 3]` | `-3.25` | `log10` number fraction |
| Mg, `metal_log_number_abundances[12 - 3]` | `-4.74` | `log10` number fraction |
| Fe, `metal_log_number_abundances[26 - 3]` | `-4.84` | `log10` number fraction |

The outputs make the precedence rule concrete:

- Mg receives the generic metal offset and the alpha offset:
  `-4.44 - 0.50 + 0.20 = -4.74`.
- O would receive both generic offsets, but the explicit `{8: +0.10}`
  entry is an absolute bracket offset relative to solar and overrides them:
  `-3.35 + 0.10 = -3.25`.
- The explicit Fe entry likewise gives `-4.54 - 0.30 = -4.84`, rather than
  applying `metallicity=-0.50`.

With the exact fixed helium number fraction `0.078370`,

```python
hydrogen_fraction + 0.078370 + (10**metal_log_number_abundances).sum()
```

returns `1.0` in this run. The chapter should say explicitly that the metal
array stores absolute base-10 logarithms of number fractions, whereas the
returned hydrogen abundance is already linear.

## 2. Direct-[X/H] vector: complete input and centidex lattice

The exact public function accepts one positional mapping,
`abundance_by_atomic_number`. It does **not** accept a sparse override mapping:
all 81 finite-solar-reference public abundances are required.

```python
import numpy as np

from payne_zero_atmosphere.direct_abundance import (
    DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX,
    DIRECT_XH_ATOMIC_NUMBERS,
    DIRECT_XH_SENTINEL_ATOMIC_NUMBERS,
    complete_direct_abundance_vector,
)

abundance_by_atomic_number = {
    atomic_number: -0.37
    for atomic_number in DIRECT_XH_ATOMIC_NUMBERS
}
abundance_by_atomic_number.update({
    8: -0.124,
    12: +0.113,
    26: -0.376,
})

realized_abundance_vector = complete_direct_abundance_vector(
    abundance_by_atomic_number
)
lattice_residual = np.max(np.abs(
    realized_abundance_vector / DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX
    - np.rint(
        realized_abundance_vector / DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX
    )
))
```

Exact results:

| Check | Result |
| --- | ---: |
| public input labels | `81` |
| returned slots, ordered by `Z=3..99` | `97` |
| O, slot `8 - 3` | `-0.12` dex |
| Mg, slot `12 - 3` | `+0.11` dex |
| Fe, slot `26 - 3` | `-0.38` dex |
| maximum centidex lattice residual | `0.0` |
| 16 sentinel slots all equal realized Fe | `True` |

The values in this mapping and returned vector are bracket abundances
`[X/H]`, not the absolute log number fractions returned by
`compute_metal_log_number_abundances`. The function first fills the 16
no-solar-reference solver slots with `[Fe/H]`, writes the 81 public values,
then quantizes the complete vector to `0.01` dex. It subsequently enforces
the exact support bounds: realized `[Fe/H]` must lie in `[-2.5, +0.5]`, and
each realized `[X/Fe]` must lie in `[-0.5, +0.5]`.

For teaching, use values away from an exact half-centidex tie, as above. The
implementation uses `np.round`; a hand-written decimal-rounding rule can
therefore create an avoidable mismatch at ties.

## 3. Fixed-column deck values to linear elemental abundances

`linear_elemental_abundances(model)` decodes a different boundary. A deck's
`fixed_column_abundance_values` stores H and He linearly but metals as base-10
log number fractions. The smallest type-honest demonstration is a one-layer
`ModelAtmosphere`; it is a representation fixture, not a physical model:

```python
import numpy as np

from payne_zero_atmosphere.atmosphere_io import (
    ModelAtmosphere,
    linear_elemental_abundances,
)
from payne_zero_atmosphere.warm_start import HELIUM_NUMBER_FRACTION

fixed_column_abundance_values = {
    1: hydrogen_fraction,
    2: HELIUM_NUMBER_FRACTION,
    **{
        atomic_number: float(
            metal_log_number_abundances[atomic_number - 3]
        )
        for atomic_number in range(3, 100)
    },
}

one = np.ones(1, dtype=np.float64)
zero = np.zeros(1, dtype=np.float64)
deck_like_model = ModelAtmosphere(
    column_mass=1.0e-4 * one,
    temperature=5772.0 * one,
    gas_pressure=1.0e3 * one,
    electron_density=1.0e12 * one,
    rosseland_opacity=0.3 * one,
    radiative_acceleration=zero.copy(),
    microturbulence=2.0e5 * one,
    convective_flux=zero.copy(),
    convective_velocity=zero.copy(),
    fixed_column_abundance_values=fixed_column_abundance_values,
)
elemental_abundances = linear_elemental_abundances(deck_like_model)
```

The returned array has 99 linear slots, ordered by `Z=1..99`. Selected exact
results are:

| Element | Linear number fraction |
| --- | ---: |
| H | `0.920874144997871` |
| He | `0.07837` |
| O | `0.0005623413251903491` |
| Mg | `1.8197008586099827e-05` |
| Fe | `1.445439770745928e-05` |

The sum is `0.9999999999999998`; taking `log10` of the O, Mg, and Fe entries
returns `-3.25`, `-4.74`, and `-4.84`. The decoder requires a complete
99-element block and raises if any slot is missing. Although its annotation
names `ModelAtmosphere`, its numerical work reads only
`model.fixed_column_abundance_values`; using the actual dataclass in the
chapter keeps the boundary honest.

## 4. Schema-v4 success and failure gallery

The exact success call is:

```python
from pathlib import Path

from payne_zero_synthesis.atmosphere import validate_atmosphere_npz

fixture_path = Path(
    "data/fixtures/chapter02_schema_v4_minimal.npz"
)
validated_names = validate_atmosphere_npz(fixture_path)
```

It returned all 25 names in `REQUIRED_ATMOSPHERE_ARRAYS`, in canonical order.
The function returns the public-name tuple; it validates by loading the whole
archive but does not return the loaded arrays.

Three copies of the fixture were modified one way at a time in a temporary
directory. The exact diagnostic suffixes were:

| Mutation | Raised diagnostic |
| --- | --- |
| remove `temperature` | `is missing required arrays: temperature` |
| set two adjacent `column_mass` values equal | `column_mass must be strictly increasing` |
| change `fractional_doppler_widths` from `(4, 6, 139)` to `(4, 6, 138)` | `partition_normalized_populations, ion_stage_populations, and fractional_doppler_widths must all have shape (n_depth, 6, 139)` |

One important limit should appear immediately after the successful examples.
Swapping the fixture's `temperature` and `gas_pressure` arrays still validates:
both are finite, positive arrays with shape `(4,)`. A structural validator can
check names, dtypes, shapes, finiteness, positivity, selected ordering, and
cross-array consistency; it cannot recover physical meaning or units from
anonymous floating-point values. This is why Chapter 2 must teach the schema
and manifest contract alongside runtime validation rather than implying that
`validate_atmosphere_npz` proves physical correctness.

## Integration traps to preserve in the canonical chapter

1. Do not place standard-mixture absolute log number fractions, direct
   `[X/H]` values, and fixed-column mixed-format values in one undifferentiated
   “abundance vector.”
2. `absolute_abundance_offsets` overrides generic metal/alpha scaling for an
   element; it is not added on top of those offsets.
3. `complete_direct_abundance_vector` is complete-input-only and experimental
   initializer infrastructure. Its 97-slot return is quantized and
   support-checked; it is not a sparse convenience API and not a converged
   atmosphere.
4. `linear_elemental_abundances` requires all 99 fixed-column entries. H and He
   are already linear; exponentiating them would be wrong.
5. The minimal schema fixture proves interface behavior only. Passing
   `validate_atmosphere_npz` does not establish that an archive is a converged
   atmosphere, that similarly shaped columns have the right units, or that its
   values are astrophysically meaningful.
