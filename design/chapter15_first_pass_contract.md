# Chapter 15 first-pass implementation contract

## Purpose and invariant

Chapter 15 answers:

> Which typed path turns stellar labels into an exploratory spectrum, and
> which independent path is allowed to claim a verified physical atmosphere?

The chapter composes prior public APIs. It adds no opacity, transfer,
correction, PCA, initializer, or synthesis kernel and invents no public result
wrapper.

The two routes remain distinct in code, provenance, tests, and prose:

```text
exploratory
labels/mixture
  -> initialize_atmosphere_from_labels
  -> InitializedAtmosphere [converged=False, closure_required=True]
  -> supported in-memory schema-v4 bridge
  -> Chapter 10 synthesis
  -> LabelSpectrum [converged=False, closure_required=True]

verified
labels/mixture
  -> solve_structured_atmosphere
  -> converged-only schema-v4 Path
  -> validate_atmosphere_npz / load metadata / load arrays
  -> Chapter 10 synthesis
  -> Spectrum + editorial acceptance rows
```

An initialized artifact may be finite, schema-valid, and spectrally useful
without being hydrostatic or radiative-convective closure. A structurally
converged solver result is still evaluated independently for flux,
hydrostatic, optical-grid, schema, and spectral acceptance.

## Pinned identities

- Payne Zero commit:
  `9c44001feae40b85146630499e6f8a5fed42e5af`
- `src/payne_zero_synthesis/api.py`:
  `77718303c1e0052a520ece7fab277b3b1922c21d09b35a288596592d03310940`
- `src/payne_zero_synthesis/atmosphere.py`:
  `06b79770e4d9472093655022d53ee7fddf7cc6727206f34c0f60c57151e2cf9b`
- `src/payne_zero_atmosphere/warm_start.py`:
  `3a83af3d68be52a35bfc3f55f5912770661be8251cc28f28da3250b2e83e0ad3`
- `src/payne_zero_atmosphere/direct_abundance.py`:
  `ec65683eb344c4c3fd77340c084e780f58c6401e77c9f0d6db05ef6753131445`
- schema v4:
  `2ba8d637e613be12ff43ce319a752616323f0341ea69f8e2391c3c244939777a`
- ordinary initializer manifest:
  `fe8093a6c0260f2524efd35a974790797e2cc8f921e9dd109477f03d845a789c`
- direct initializer manifest:
  `fb59da5e6bd3f8fcba06e0c4c284137e90aab5c4e93165daa74d8ce2ae268710`
- `five_label`, `cno8`, and `direct_abundance` checkpoint hashes:
  `c3271701…`, `be97c0b7…`, and `1b8e1db1…`.

The verifier recomputes these identities locally. External source trees are
comparison-only and never a runtime dependency.

## Exact public transformations

### Exploratory boundary

The exact public types are:

```text
ForwardTimings
    initializer_seconds
    population_bridge_seconds
    synthesis_seconds
    total_seconds

InitializedAtmosphere
    structured_atmosphere
    initializer_family
    labels
    provenance
    timings
    atmosphere_converged = False
    atmosphere_closure_required = True

LabelSpectrum(Spectrum)
    wavelength_nm
    flux_total
    flux_continuum
    normalized_flux
    seconds
    initializer_family
    labels
    provenance
    timings
    initialized_atmosphere
    atmosphere_converged = False
    atmosphere_closure_required = True
```

`initialize_atmosphere_from_labels` preserves requested physical labels while
recording family selection, projections, sparse direct-abundance completion,
asset identities, and limitations in provenance. `synthesize_from_labels`
passes the exact in-memory structured mapping to the Chapter 10 synthesis
engine and returns `LabelSpectrum`.

`InitializedAtmosphere.save_npz` may persist an initializer-marked schema-v4
artifact with:

```text
atmosphere_product_role = learned_initializer_prediction
atmosphere_converged = False
atmosphere_closure_required = True
```

This is not physical-product promotion.

### Verified boundary

`solve_structured_atmosphere` returns only the promoted
`payne_zero_structured_atmosphere.npz` `Path`. It tries deterministic
ordinary/CNO candidates in order or one complete direct-abundance trial. It
does not return a failed path: exhaustion raises.

The exact downstream sequence is:

```text
Path
  -> validate_atmosphere_npz(path): tuple[str, ...]
  -> load_atmosphere_product_metadata(path): dict | None
  -> load_atmosphere_npz(path): dict[str, np.ndarray]
  -> synthesize(path or mapping): Spectrum
```

Optional product metadata may validly be absent. Schema-v4 validation is the
required gate.

## Labels and mixture preservation

Four capstone requests are immutable typed records:

1. hot dwarf — ordinary five-label, atomic synthesis;
2. solar dwarf — ordinary five-label;
3. low-gravity giant — ordinary five-label with non-solar metallicity and
   alpha enhancement;
4. cool molecule-rich — CNO-aware with explicit C/N/O coordinates.

Every record carries:

- `effective_temperature`, `log_surface_gravity`, `metallicity`,
  `alpha_enhancement`, and `microturbulence_km_s`;
- explicit C/N/O values or explicit absence;
- requested initializer family;
- molecular-line policy;
- one canonical JSON identity and SHA-256.

Family routing may change only the initializer query. It cannot alter the
requested physical labels or mixture supplied to the exact solver.

Direct-abundance mode is a separate experimental route. Sparse public
coordinates inherit the declared `[Fe/H]` baseline at the high-level boundary;
the 97-slot exact mixture, lattice quantization, manifest status, and
experimental limitation remain provenance. Unsupported direct inputs reject;
they never fall back to a family that erases mixture detail.

## Schema-v4 and Chapter 10 handoff

The chapter validates all 25 canonical public arrays, exact depth-axis
agreement, float64 public columns, positive/increasing column mass, positive
temperature/pressure/density/opacity fields, and schema version 4.

The synthesis handoff uses exact Chapter 10 names:

```text
Spectrum.wavelength_nm
Spectrum.flux_total
Spectrum.flux_continuum
Spectrum.normalized_flux
Spectrum.seconds
```

`normalized_flux == flux_total / flux_continuum` is checked after wavelength
alignment. Device, dtype, resolution, catalog view, cache policy, and source
hashes are recorded separately from physical labels.

## Editorial acceptance rows

The acceptance table is a chapter-local tuple of typed rows, not a public
workflow class. Gates are independent:

1. initializer support and asset integrity;
2. seed finite/positive/monotone;
3. structural convergence;
4. flux-error acceptance;
5. hydrostatic residual acceptance;
6. optical-depth/grid acceptance;
7. schema-v4 validation;
8. trajectory parity when a golden exists;
9. wavelength and three spectral-array parity gates;
10. backend/cache/thread/data identity;
11. supported-physics and limitation flags.

Each row has `passed: bool | None`; `None` means not applicable or blocked,
never silently passed. Exploratory rows must set structural, flux, and
hydrostatic acceptance to `None`.

The four-regime table reports route, family, molecules, schema, spectral
finiteness, closure status, and the exact blocking reason. Compact Chapter 10
fixtures are integration evidence, not publication-grade four-regime
atmosphere closure or golden-spectrum parity.

## Reproducibility

One canonical provenance digest includes:

- pinned commit;
- request digest;
- source, schema, initializer asset, and synthesis-input hashes;
- Python, NumPy, and Torch versions;
- device and dtype;
- fixed atmosphere-thread policy where applicable;
- synthesis resolution and wavelength interval;
- cache policy and catalog identity;
- exact availability/blocker status.

Canonical JSON uses sorted keys and compact separators. Timing values are
reported but excluded from identity. Repeated digest construction must be
bitwise identical.

## Executable artifacts

`book/chapter15_teaching.py` owns only transparent composition:

- immutable label/mixture request records and canonical digests;
- independent acceptance rows;
- atmosphere schema/shape/positivity summaries;
- spectral ratio and wavelength-alignment summaries;
- provenance normalization and digesting;
- workflow-boundary status formatting.

`book/chapter15_runtime.py` owns:

- exact source/asset identity verification;
- public type/signature checkpoints;
- four immutable regime requests;
- exact Chapter 10 compact schema-v4/spectrum checkpoints;
- exploratory safety-flag and initializer-marked persistence checks where the
  public initializer seam is callable;
- verified-path availability probing without starting an expensive solve;
- four-regime capstone and reproducibility checkpoints;
- explicit blocker records.

`book/chapters/chapter_15.py` is a concise capstone. It routes and calls; it
does not reproduce prior kernels. Tests validate types, hashes, labels,
mixtures, schema, spectral semantics, independent gates, and blocker honesty.

No new golden physical atmosphere or spectrum is manufactured while the exact
runner seam is absent.

## Current dependency status

At contract creation:

- exact Chapter 10 synthesis source and compact four-regime fixtures are
  available;
- all three Chapter 14 checkpoint assets and exact initializer source modules
  are available;
- the staged atmosphere package does not yet export the complete Chapter 14
  public initializer surface required by `initialize_atmosphere_from_labels`;
- the staged runner lacks all eight Chapter 13 orchestration symbols,
  including `run_atmosphere_model`;
- `solve_structured_atmosphere` is not staged.

Therefore the complete first-pass spine executes:

- exact type/signature/source identity;
- exact schema-v4 load/validation semantics;
- exact Chapter 10 four-regime compact synthesis;
- labels/mixture/provenance and acceptance composition;
- safety and failure-boundary checks.

It marks exploratory initializer execution `initializer_export_seam_blocked`
until Chapter 14 exports land, and verified physical execution
`exact_runner_seam_blocked` until Chapter 13 orchestration and the public
solver entry point land. It never substitutes a Chapter 10 fixture for either
an initializer prediction or a converged physical product.

## End contract

Chapter 15 yields one honest choice:

- a fast `InitializedAtmosphere`/`LabelSpectrum`, explicitly unconverged and
  closure-required; or
- a converged-only `Path`, validated schema-v4 mapping, and exact `Spectrum`
  accompanied by independent editorial acceptance rows.

If a required seam or full-data artifact is unavailable, the corresponding
gate is blocked and named. No wrapper, prose claim, or attractive spectrum may
broaden that status.
