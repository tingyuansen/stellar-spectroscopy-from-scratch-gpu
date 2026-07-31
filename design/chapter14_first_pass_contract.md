# Chapter 14 first-pass contract

## Purpose

Chapter 14 answers one question:

> Can a learned model place the exact atmosphere iteration inside its
> convergence basin without being confused with the converged fixed point?

The chapter builds a starting atmosphere from labels, then preserves the
boundary

\[
\mathbf{x}^{(0)}_{\rm init}\ne \mathbf{x}_\star,\qquad
\mathbf{x}_\star=\mathcal G(\mathbf{x}_\star).
\]

The initializer proposes \(\mathbf{x}^{(0)}\). The Chapter 13 iteration
\(\mathcal G\) remains the only closure test. No decoded profile is called a
converged atmosphere or used to write a physical product.

## Pinned implementation

- Payne Zero commit:
  `9c44001feae40b85146630499e6f8a5fed42e5af`
- `src/payne_zero_atmosphere/warm_start.py`:
  `3a83af3d68be52a35bfc3f55f5912770661be8251cc28f28da3250b2e83e0ad3`
- `src/payne_zero_atmosphere/direct_abundance.py`:
  `ec65683eb344c4c3fd77340c084e780f58c6401e77c9f0d6db05ef6753131445`

Both staged modules are byte-identical to the pinned source. Chapter 14 may
wrap them for teaching and verification, but must not fork their algorithms
or rename their public quantities.

The runtime assets are:

| family | role | SHA-256 |
|---|---|---|
| `five_label` | ordinary five-feature initializer | `c32717016d4f9047ab37bc17b6900faf9c514de3407292a07a745c35a50784e5` |
| `cno8` | C/N/O-aware eight-feature initializer | `be97c0b729490b9e18f092c59f836adb421e2e99359b44436edc3ae5632f8a02` |
| `direct_abundance` | experimental 84-feature initializer | `1b8e1db1514956dfbf890eb5ae96e01bd918acfc86be538b6b77230332104243` |

The ordinary/CNO release manifest and the direct experimental manifest are
packaged. Training corpora are not runtime dependencies and are not copied.

## Act I — one representation and one decoder

### Six physical profiles

The public prediction is `dict[str, np.ndarray]`. Its exact keys and units
are:

| field | unit | condition |
|---|---|---|
| `column_mass` | g cm\(^{-2}\) | positive, strictly increasing |
| `temperature` | K | positive |
| `gas_pressure` | dyn cm\(^{-2}\) | positive |
| `electron_density` | cm\(^{-3}\) | positive |
| `rosseland_opacity` | cm\(^2\) g\(^{-1}\) | positive |
| `radiative_acceleration` | cm s\(^{-2}\) | signed |

Every array is NumPy `float64[80]`.

The checkpoint coordinate order is exact:

1. `log10_column_mass_increment`
2. `log10_temperature_relative_to_grey`
3. `log10_gas_pressure`
4. `log10_electron_density`
5. `log10_rosseland_opacity`
6. `asinh_radiative_acceleration`

With \(\Delta m_0=m_0\), \(\Delta m_j=m_j-m_{j-1}\), and

\[
T_{{\rm grey},j}=T_{\rm eff}
\left[\frac34\left(\tau_{{\rm R},j}+\frac23\right)\right]^{1/4},
\]

the six forward coordinates are

\[
\left(
\log_{10}\Delta m,\,
\log_{10}(T/T_{\rm grey}),\,
\log_{10}P_{\rm gas},\,
\log_{10}n_e,\,
\log_{10}\kappa_{\rm R},\,
\operatorname{asinh}(g_{\rm rad}/s_g)
\right).
\]

They form `(80, 6)` and flatten in C order to 480 values. The inverse used
at runtime is exactly:

```text
column_mass = cumsum(10**clip(u[:, 0], -30, 30))
temperature = grey_temperature * 10**clip(u[:, 1], -3, 3)
gas_pressure, electron_density, rosseland_opacity
    = 10**clip(u[:, 2:5], -30, 30)
radiative_acceleration
    = acceleration_scale * sinh(clip(u[:, 5], -20, 20))
```

### PCA and dtype boundary

The MLP returns 160 standardized PCA coefficients. Its one intentional
device/dtype boundary is:

1. construct and standardize manifest-ordered labels in NumPy;
2. create one Torch `float32` batch on the selected device;
3. run the SiLU network under `torch.no_grad()`;
4. detach and cross once to CPU NumPy;
5. de-standardize coefficients in NumPy `float64`;
6. compute `coefficients @ basis`, where `basis.shape == (160, 480)`;
7. de-standardize 480 coordinates in NumPy `float64`;
8. reshape to `(80, 6)` in C order;
9. invert the six transforms.

This boundary is tested independently of deck formatting. A sentinel
coefficient test pins the `160 -> 480 -> (80, 6)` orientation.

### Fixed-column quantization

The decoded arrays are only floating predictions. The exact solver seed is
formed by:

1. packing the nine-column layer table;
2. formatting it with `format_warm_start_deck`;
3. parsing that text with `parse_atmosphere_deck`;
4. retaining both `ModelAtmosphere` and the identical `deck_text`.

This format/parse is deliberate quantization, not serialization overhead.
Every initializer family crosses it exactly once before its first exact pass.

## Act II — ordinary and CNO-aware starts

### Feature order

`five_label` uses:

```text
temperature_ratio_5040_k_over_temperature
log10_surface_gravity_cgs
metallicity
alpha_enhancement
microturbulence_km_s
```

`cno8` appends:

```text
carbon_enhancement
nitrogen_enhancement
oxygen_enhancement
```

The public physical labels keep the source notation:
\(T_{\rm eff}\), \(\log g\), \([\mathrm M/\mathrm H]\),
\([\alpha/\mathrm M]\), \(\xi\), and optionally
\([\mathrm C/\mathrm M]\), \([\mathrm N/\mathrm M]\),
\([\mathrm O/\mathrm M]\).

### Routing

`select_warm_start_family` chooses `cno8` if any relative C/N/O label is
explicit, or if an absolute abundance override contains atomic number 6, 7,
or 8. Otherwise it chooses `five_label`.

There is no cross-family fallback. An explicit CNO pattern must never be
silently discarded.

Missing CNO coordinates in the eight-label family resolve as:

```text
[C/M] = 0
[N/M] = 0
[O/M] = [alpha/M]
```

An absolute `[X/H]` override resolves to `[X/M] = [X/H] - [M/H]`; duplicate
absolute and relative input must agree.

### Support and deterministic candidates

Ordinary/CNO support projection changes only the initializer query. The
requested labels and abundances written into the seed deck—and later passed
to the exact solve—remain unchanged.

`deterministic_initializer_labels` returns an ordered tuple:

- entry 0 is `None` when the request is already supported;
- otherwise entry 0 is the nearest bound-projected initializer label;
- later entries are reproducible checkpoint-width jitters;
- every jitter component is derived from SHA-256 of the canonical physical
  request, public seed, trial index, and checkpoint feature name;
- defaults are `seed=20260713`, `jitter_scale=0.01`;
- the high-level workflow may request two trials.

Candidates are tried in order. They are never averaged or selected using an
unverified spectrum.

## Act III — experimental direct abundances

### The 81/84/97 contract

- Public low-level input: 81 finite-solar-reference `[X/H]` values in
  increasing atomic number.
- Network input: 84 values:
  \(5040/T_{\rm eff}\), \(\log g\), \(\xi\), `[Fe/H]`, then 80 non-Fe
  `[X/Fe]` values in increasing atomic number.
- Exact solver/synthesis mixture: `float64[97]` for atomic numbers 3 through
  99.
- The 16 no-solar-reference sentinel slots inherit the quantized iron value.

`complete_direct_abundance_vector` is strict: missing, unknown, noninteger,
or nonfinite public inputs raise. Sparse completion belongs only at a higher
API boundary.

All abundances are placed on the exact 0.01-dex lattice with NumPy rounding
before support checks, network evaluation, caching, deck construction, or
hashing. Signed zero is canonicalized.

Support is exact:

```text
effective_temperature:                 4000 .. 10500 K
log_surface_gravity:                   0.7 .. 5.3
microturbulence_km_s:                  0.5 .. 4.0
iron_abundance_relative_to_hydrogen:  -2.5 .. 0.5 dex
element_abundance_relative_to_iron:   -0.5 .. 0.5 dex
```

Unsupported direct inputs raise. They are never projected and never routed
to five-label or CNO models.

### Set encoder

The first four stellar-state features are shared by all element tokens. Each
of the 80 non-Fe elements contributes:

- its learned element embedding;
- its abundance amplitude \(a=[X/Fe]\);
- \(a^2\);
- linear and quadratic response vectors produced by the same SiLU response
  law.

The 80 response vectors are summed, concatenated with the four state
features, and decoded to the shared 160 PCA coefficients. Element identity
and abundance must move together; permuting abundances alone is not an
invariance.

### Safety and provenance

All public direct entry points default opt-in flags to `False`. The retained
checkpoint has `release_gate_passed=False`.

The decoded optimizer object is
`DirectAbundanceOptimizerSurrogate`. Its fixed safety fields are:

```text
role = "experimental_direct_xh_optimizer_surrogate"
exact_closure_required = True
is_final_atmosphere = False
```

It carries immutable realized 97-slot abundances plus checkpoint, manifest,
mixture, deck, and surrogate-identity hashes.

`direct_abundance_warm_start_deck` returns only a provenance-marked `str`.
It intentionally does not expose the parsed starting `ModelAtmosphere`.

`run_direct_abundance_atmosphere` is the only direct route to an
`AtmosphereRunResult`. It requires:

- explicit experimental opt-in;
- convergence stopping enabled;
- enough passes for the configured minimum/consecutive rule;
- one exact trial using the same quantized 97-slot mixture;
- structural convergence;
- terminal mixture equality.

Failure returns no physical result.

## Executable chapter artifacts

`book/chapter14_teaching.py` owns transparent calculations:

- six forward/inverse profile transforms;
- PCA sentinel/orientation trace;
- dtype-boundary trace;
- family-routing examples;
- deterministic-candidate summaries;
- direct layout, lattice, sentinel, hash, and set-token traces.

`book/chapter14_runtime.py` owns exact checkpoints against the staged source
and local assets:

- source and asset identity;
- ordinary/CNO checkpoint schema and decoder;
- five-label and CNO quantized seeds;
- direct opt-in, manifest provenance, layout, support, and decoded surrogate;
- the explicit shared-runner seam status.

`book/chapters/chapter_14.py` contains the causal, bite-sized narrative. It
uses short executable cells and mostly one-panel plots. It ends with a
summary and a link to Chapter 15. There is no detached exercise section and
no large Python block inside Markdown.

The fixture/golden bundle contains only small inputs and selected coefficients
or decoded/quantized outputs. The checkpoints themselves are immutable static
runtime assets. The verifier recomputes before comparison.

## Parity gates

1. Staged source and all packaged checkpoint/manifest hashes are exact.
2. All six transform round trips satisfy declared tolerances.
3. Torch output is `float32`; all PCA and physical decoding arrays are
   NumPy `float64`.
4. Five/CNO feature order, routing, projection, and deterministic candidate
   order match source.
5. Five/CNO quantized seeds match pinned goldens.
6. Direct 81/84/97 sizes, atomic-number order, 16 sentinels, 0.01-dex lattice,
   and mixture hash match source.
7. Direct opt-in, release-gate, unsupported-input rejection, and immutable
   surrogate flags are enforced.
8. Direct quantized seed matches its pinned golden.
9. No initialized object can claim convergence or write the physical product
   schema.

## Current shared-runner boundary

The staged Chapter 13 shared runner seam is not yet complete. Chapter 14 may
execute all initializer, decoder, quantization, and provenance gates. Exact
restart trajectory and terminal closure gates remain explicitly
`seam_blocked` until `run_atmosphere_model` and its required finalization
records exist in the shared staged runner.

This is not Chapter 13 closure and must not be described as such.

## End contract

The reader leaves Chapter 14 able to build an exact ordinary, CNO-aware, or
experimental direct-abundance starting state from first principles. They can
explain every transform, dimension, dtype crossing, dispatch decision,
quantization step, and safety gate. They also know precisely why that starting
state is a proposal—not a physical atmosphere—until the exact fixed-point
iteration converges.
