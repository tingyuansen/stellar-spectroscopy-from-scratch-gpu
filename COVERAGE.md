# Payne Zero Coverage Ledger

This ledger prevents the book from losing a physics branch, numerical contract, data boundary,
or optimization while removing redundancy. The pinned source identities are in `PLAN.md`.

Status vocabulary:

- `planned` — chapter and gate assigned;
- `implemented` — canonical textbook code and prose exist;
- `verified` — the assigned Payne Zero parity gate passes;
- `integrated` — neighbor, redundancy, notation, render, and whole-book reviews pass;
- `boundary` — intentionally not reimplemented; exact reason and interface are taught.

No item is complete before `integrated`.

Evidence snapshot for this reconciliation:

- all fifteen canonical chapter sources exist, and all fifteen checked-in
  notebooks/HTML files contain executed code with no stored error outputs;
- a fresh source-to-notebook audit after the first whole-book flow pass reports
  zero cell-source drift across Chapters 1–15;
- Chapter 12 source and Chapter 13–15 artifact verifiers pass, and the global
  pinned-fragment verifier matches all 58 staged modules to commit
  `9c44001feae40b85146630499e6f8a5fed42e5af`;
- Chapter 6's complete focused suite now reports 215 passed and 1 skipped;
  both its 19-array atmosphere and 213-member synthesis products pass
  read-only exact publication verification;
- the Chapter 10/15 runtime now rebinds already imported staged path defaults
  to its disposable role-safe data view; isolated and representative
  import-order regression suites pass;
- one exact 80-layer solar atmosphere converges on pass four, and staged,
  staged-repeat, and pinned-read-only runs produce byte-identical 27-array
  atmosphere archives;
- the corresponding wavelength, total-flux, continuum-flux, and
  normalized-flux arrays are bitwise-identical; wall-clock timing is excluded
  from the comparison product;
- flux-error, hydrostatic-residual, and separately retained
  standard-optical-grid acceptance for that solar product remain explicitly
  unevaluated;
- the full repository suite will be rerun after the concurrent symbol and
  Chapter 10 prose revisions are integrated.

## Chapter Acceptance

| Chapter | Construction status | Independent acceptance | Whole-book status |
| --- | --- | --- | --- |
| 1. From Starlight to a First Grey Atmosphere | executed and rendered | accepted, no open P0/P1 | retained for final whole-book audit |
| 2. From Equations to Fast, Trustworthy Kernels and Explicit Data | executed and rendered | accepted, all P1 closed | first whole-book flow audit passed; retained for physics/polish audit |
| 3. Atoms, Ions, and Electrons | executed and rendered | accepted, no open P0/P1 | retained for final whole-book audit |
| 4. Molecules and Coupled Equilibrium | executed and rendered | accepted, no open P0/P1; 141-test integrated gate | retained for final whole-book audit |
| 5. Continuous Opacity and Scattering | published, executed, and rendered | accepted, no open P0/P1; manifest re-audit accepted; 321-test repository gate | retained for final whole-book audit |
| 6. One Spectral Line | published, executed, and rendered | 215 passed, 1 skipped; exact read-only verification of the 19-array atmosphere and 213-member synthesis products | first neighbor-flow audit passed; retained for final physics/polish audit |
| 7. Atomic Line Forests and Special Profiles | published, executed, and rendered | 10 compact-catalog/profile/deposit/runtime gates pass | first neighbor-flow audit passed; retained for physics/polish audit |
| 8. Molecular Bands and Source Compilation | published, executed, and rendered | 12 parser/compiler/cache/family-boundary/runtime gates pass | first neighbor-flow audit passed; retained for physics/polish audit |
| 9. Radiative Transfer with Scattering | published, executed, and rendered | 11 atmosphere/synthesis transfer, boundary, flux, and chunk-policy gates pass | first neighbor-flow audit passed; retained for physics/polish audit |
| 10. GPU Synthesis from a Structured Atmosphere | published, executed, and rendered | compact/public CPU-float64 pipeline and import-order-safe role-bound data view verified | first flow audit passed; pedagogical deepening in progress |
| 11. Starting and Blanketing an Atmosphere | published, executed, and rendered | 6 exact 80-layer population/opacity/selection/reuse gates pass | exact runtime and rendered source are synchronized; first neighbor-flow audit passed |
| 12. Radiation, Thermodynamics, and Convection | published, executed, and rendered | 10 gates pass, including the pinned-source verifier and exact Ch. 11 `OpacityState` → `IterationFinalization` handoff | exact runtime and rendered source are synchronized; first neighbor-flow audit passed |
| 13. Correction and the Full Numba Iteration | published, executed, and rendered | component/remap gates pass; the exact solar trajectory reaches the declared deep-temperature criterion on pass four | verified iteration spine and one accepted structural trajectory |
| 14. Learned Initializers and Mandatory Physical Closure | published, executed, and rendered | initializer/direct-abundance gates pass; the ordinary solar initializer closes through the exact solver | verified initializer boundary and one exact closure route |
| 15. From Stellar Labels to a Verified Spectrum | published, executed, and rendered | exact public exploratory call plus three-run solar atmosphere/spectrum parity; 24 focused capstone/manifest tests pass | first flow audit passed; solar structural/schema/spectrum/parity evidence integrated, with three independent physical gates still unevaluated |

### Open acceptance blockers

1. Reconcile conservative symbol statuses against the completed chapter and
   exact-test evidence.
2. Run and accept full physical trajectories for the hot dwarf,
   low-gravity giant, and cool molecule-rich capstone requests.
3. Retain and review independent flux-error, hydrostatic-residual, and
   standard-optical-grid acceptance evidence for physical-product claims.
4. Complete optional cross-backend and cross-thread tolerance evidence.
5. Repeat the full repository suite, rendered visual audit, and final
   repository-cleanliness pass after the active revisions land.

## Cross-Cutting Contracts

| Contract | Primary location | Gate | Status |
| --- | --- | --- | --- |
| 1D plane-parallel LTE scope | Ch. 1, 15 | scope audit | implemented |
| CPU atmosphere / GPU synthesis / Torch initializer split | Ch. 1–2, 10, 13–15 | architecture audit | verified |
| outer-to-inner depth order | Ch. 1–2 | schema tests | verified |
| actual versus partition-normalized populations | Ch. 2–3 | schema + identity tests | integrated |
| atmosphere packed `(D,1006)` state | Ch. 3 | layout tests | integrated |
| synthesis `(D,6,139)` state | Ch. 2–3 | schema tests | integrated |
| fixed-ne synthesis bridge versus full charge solve | Ch. 3–4, 10 | bridge parity | integrated |
| exact versus parity-pinned constants | Ch. 2, App. A | constant identity | verified |
| fixed-column quantization | Ch. 11, 13 | format/parse trajectory | verified |
| static/subset/fixture/golden data roles | Ch. 2, App. B | manifest audit | verified |
| cold versus warm cache timing | Ch. 2, 10, 13 | cache equality/timing | verified |
| fixed-thread versus cross-thread reductions | Ch. 7, 11–13 | thread policy | implemented |
| `H_nu`/`F_nu`/`F_lambda` conversion | Ch. 9–10 | output identity | verified |
| initialized versus converged atmosphere role | Ch. 1, 14–15 | metadata/closure | implemented |
| exact intrinsic-grid spelling by interface | Ch. 10, 15, App. C | signature/CLI/cache audit | verified |
| atmosphere/synthesis molecular line-family status | Ch. 7–8, 11 | two-lane feature matrix | verified |
| H2O compiler/runtime status | Ch. 8, 15 | feature matrix | verified |
| direct-abundance experimental status | Ch. 14–15 | safety metadata | verified |

## Exact Intrinsic-Grid Interface Matrix

The physical quantity is always the intrinsic logarithmic grid density
\(R_{\rm grid}=\lambda/\Delta\lambda\), never instrumental resolving power.
Exact source spelling is interface-specific:

| Boundary | Pinned spelling and behavior | Book location |
| --- | --- | --- |
| mathematics | \(R_{\rm grid}\) | Ch. 6–10 |
| `synthesize_from_labels` | canonical `r_grid`; `resolution` is accepted as an alias; unequal double specification is rejected | Ch. 15, App. C |
| public archive/in-memory `synthesize` | `resolution` only | Ch. 10, 15, App. C |
| `SynthesisPipeline`, `Grid`, engine/window helpers, molecular compilers, and cache records | `resolution` | Ch. 7–10, App. B/C |
| synthesis and prewarm CLIs | `--r-grid` and `--resolution` aliases with `dest="resolution"` | Ch. 15, App. C |

## Two-Lane Line-Family and Reuse Matrix

“Default” below means the standard high-level workflow at the pinned source
commit, not the all-`None` defaults of the low-level `AtmosphereInput`
dataclass.

| Family or lifecycle | Atmosphere lane | Synthesis lane | Ownership |
| --- | --- | --- | --- |
| ordinary atomic | default runtime: predicted, observed, and high-excitation source arrays are selected into `SelectedLineCatalog` and deposited | default runtime, independent of `molecular_lines` | Ch. 7 teaches routes/common deposit; Ch. 11 composes atmosphere selection |
| diatomic / text bands | default runtime through `diatomic_lines_path`, family correction, then common selected-line deposit | default-on runtime when `molecular_lines=True`; manifest-ordered text compiler | Ch. 8 family rules; Ch. 11 atmosphere composition |
| TiO | default runtime through `titanium_oxide_lines_path` and common selected-line deposit | default-on runtime when `molecular_lines=True`; Schwenke compiler | Ch. 8; Ch. 11 atmosphere composition |
| water / H2O | default runtime through `water_lines_path`, the dedicated water selector, and common selected-line deposit | compiler-only: `compile_h2o_partridge` exists, but standard `_compile_molecular` omits it | Ch. 8 exact cross-lane boundary; Ch. 11 active atmosphere composition |
| H3+ | opt-in runtime: explicit `AtmosphereInput.h3plus_lines_path`; no file is returned by `source_line_paths()` | absent from the standard source compiler and pipeline; a mass-table entry is not runtime wiring | Ch. 8 optional/absent boundary; Ch. 11 may compose the atmosphere option |
| selected/detailed-object reuse | first opacity pass generates or loads one common selected catalog and, when enabled, loads one detailed-transition catalog; later atmosphere iterations reuse those objects | atomic and enabled molecular `WindowInvariants` are cacheable/reusable by exact physical key | Chs. 7–8 own contents; Ch. 11 owns atmosphere select-once/reuse; Ch. 10 owns synthesis cache composition |

## Atmosphere Package Modules

| Module | Non-redundant responsibility | Primary chapters | Gate | Status |
| --- | --- | --- | --- | --- |
| `__init__.py` | public export surface | App. C/D | API inventory | verified |
| `__main__.py` | module CLI entry | App. C | CLI smoke | verified |
| `_numba_cache.py` | stable cache resolution | Ch. 2, 13 | cache path/fingerprint | verified |
| `atmosphere_io.py` | `ModelAtmosphere`, fixed-column compatibility and quantization | Ch. 2, 11 | round trip + trajectory | verified |
| `cli.py` | high-level solve, family selection, retries, atomic publication | Ch. 13, 15, App. C | workflow integration | implemented |
| `config.py` | structured inputs, outputs, controls | Ch. 11, 13 | config validation | verified |
| `constants.py` | exact and parity constant tiers | Ch. 2, App. A | byte/value identity | verified |
| `continuum_opacity.py` | full atmosphere continuum, scattering, sampling grid, Rosseland lookup | Ch. 5, 11 | component + slab parity | verified |
| `convection.py` | finite-difference thermodynamics and mixing length | Ch. 12 | derivative/flux parity | verified |
| `convergence.py` | structural stopping diagnostics | Ch. 13 | state-machine/threshold | verified |
| `data_files.py` | runtime data resolution | Ch. 2, App. B | path/manifest | verified |
| `direct_abundance.py` | 81-abundance initializer and safety/provenance | Ch. 14 | decode/restart/safety | verified |
| `doppler.py` | fractional widths and selection strength state | Ch. 3, 6 | width/factor parity | verified |
| `equation_of_state.py` | partition functions, Saha, electron closure, populations | Ch. 3 | per-layer + batch parity | integrated |
| `hydrogen_line_profile.py` | atmosphere HPROF/Stark evaluator | Ch. 7, 11 | profile/deposit parity | verified |
| `hydrostatic.py` | pressure update | Ch. 1, 11 | pressure integration | verified |
| `install_runtime_data.py` | manifest-bound data installation | App. B/C | dry-run/verification | planned |
| `line_catalog.py` | selected/detailed binary catalog decoding | Ch. 7, 11 | record identity | verified |
| `line_opacity.py` | selected/detailed/special line accumulation | Ch. 6–7, 11 | component/slab/thread | verified |
| `line_profile_math.py` | exponential, Voigt, selector, and profile tables | Ch. 6–7 | scalar/compiled parity | verified |
| `line_selection.py` | resident source selection and fused keep tests | Ch. 7, 11 | selected identity/count | verified |
| `microturbulence.py` | standard microturbulence profile | Ch. 3, 11 | profile parity | verified |
| `molecular_data.py` | atmosphere molecular catalog | Ch. 4, App. B | catalog identity | integrated |
| `molecular_equilibrium.py` | depth-continuation Newton chemistry and molecular energy | Ch. 4, 12 | state/energy parity | integrated |
| `population_layout.py` | 1006-slot mapping and job schedule | Ch. 3 | exhaustive layout | integrated |
| `prewarm.py` | cache fingerprint, representative branches, fresh-process verification | Ch. 13, App. C | prewarm manifest | implemented |
| `radiative_pressure.py` | pressure and acceleration accumulation | Ch. 12 | accumulator parity | verified |
| `radiative_transfer.py` | depth integration, differentiation, remap, tables | Ch. 9, 11–12 | helper parity | verified |
| `rosseland_mean.py` | harmonic mean and Rosseland depth | Ch. 5, 12 | mean/depth parity | verified |
| `run_setup.py` | seed and branch validation | Ch. 11, 13 | setup validation | verified |
| `runner.py` | exact physical iteration and final product | Ch. 11–13, 15 | one-pass verified; converged trajectory pending | implemented |
| `runtime_state.py` | initial density, abundance, and packed state | Ch. 2–3 | state identity | integrated |
| `source_catalogs.py` | catalog root, shards, checksums, validation | Ch. 2, 7, 11, App. B | checksum/subset audit | verified |
| `specific_internal_energy.py` | atomic internal energy | Ch. 3, 12 | energy parity | verified |
| `synthesis_bridge.py` | final schema-v4 rebuild and publication | Ch. 2, 10, 13, 15 | schema boundary; converged publication pending | implemented |
| `temperature_correction.py` | radiative/convective temperature and mass correction | Ch. 13 | component + full correction | verified |
| `transfer_kernels.py` | compiled transfer/source iteration and parallel frequency reduction | Ch. 9, 12–13 | scalar/compiled/fixed-thread | verified |
| `warm_start.py` | five-label/CNO8 transforms, network, decoder, support/retries | Ch. 11, 14 | checkpoint/decode/restart | verified |

### Atmosphere Branch Boundaries

| Branch | Treatment | Location | Status |
| --- | --- | --- | --- |
| turbulent pressure | unsupported by pinned exact runner | Ch. 1, 11, 13, 15 | boundary |
| NLTE/departure-coefficient modes | unsupported scope | Ch. 1, 7, 15 | boundary |
| HLINOP hydrogen-wing branch | unsupported by pinned exact runner | Ch. 7, 13 | boundary |
| converted diatomic/TiO/water selectors | active standard high-level runtime paths, not unsupported branches | Ch. 8, 11 | verified |
| atmosphere H3+ selector | supported only with explicit `h3plus_lines_path`; absent from `source_line_paths()` defaults | Ch. 8, 11 | boundary |
| cool high-gravity fixed-point instability | declared failure regime; no accepted full trajectory yet | Ch. 13, 15 | implemented |
| `AtmosphereOutput.diagnostics_path` | declared but unwritten upstream | App. C | boundary |

## Synthesis Package Modules

| Module | Non-redundant responsibility | Primary chapters | Gate | Status |
| --- | --- | --- | --- | --- |
| `__init__.py` | public export surface | App. C/D | API inventory | verified |
| `api.py` | public dataclasses, two workflows, flux conversion, serialization | Ch. 1, 10, 14–15 | API/types/compact workflow | verified |
| `atmosphere.py` | schema-v4 load/validate and compatibility input | Ch. 2, 10 | schema/upgrade tests | verified |
| `atomic_lines.py` | source decoding, corrections, routing, margins, typed catalog | Ch. 7, 10 | catalog identity | verified |
| `cli.py` | label and archive synthesis commands | Ch. 15, App. C | CLI smoke; full products pending | implemented |
| `constants.py` | exact and parity constant tiers | Ch. 2, App. A | byte/value identity | verified |
| `continuum.py` | synthesis continuum populations, edge sampling, interpolation | Ch. 5, 10 | component + slab parity | verified |
| `device.py` | CUDA/MPS/CPU and dtype policy | Ch. 2, 10 | backend policy | verified |
| `equation_of_state.py` | Torch Saha/populations, full-ne and fixed-ne states | Ch. 3–4, 10 | state/bridge parity | integrated |
| `ground_partition_table.py` | explicit ground-state partition corrections | Ch. 3 | table/function parity | verified |
| `hydrogen_lines.py` | fine structure, Stark/impact, merging, pseudo-continuum | Ch. 7, 10 | profile/deposit parity | verified |
| `line_opacity.py` | GPU atomic/helium/special opacity and sparse deposits | Ch. 6–7, 10 | line/backend parity | verified |
| `molecular_equilibrium.py` | 190-species Torch chemistry and stage-5 packing | Ch. 4 | state/metadata parity | integrated |
| `molecular_lines.py` | GPU molecular catalog/invariants/chunked deposits | Ch. 8, 10 | component/backend parity | verified |
| `paths.py` | runtime tables, catalogs, and cache roots | Ch. 2, 10, App. B | isolated path identity; mixed-order isolation pending | implemented |
| `pipeline.py` | window invariants, star state, complete device pipeline | Ch. 7–8, 10 | stage + compact spectrum | verified |
| `prewarm.py` | window cache preparation and provenance | Ch. 10, App. C | cold/warm equality | verified |
| `radiative_transfer.py` | batched total/continuum scattering transfer | Ch. 9–10 | transfer/backend parity | verified |
| `source_catalog_molecular_compiler.py` | text/TiO/H2O compilation and cached serial Numba | Ch. 8, App. B | scalar/Numba exactness | verified |
| `synthesis.py` | engine boundary for structured state and spectra | Ch. 10, 15 | compact/public integration | verified |

### Synthesis Feature Matrix

| Feature | Pinned capability | Book requirement | Status |
| --- | --- | --- | --- |
| ordinary atomic lines | standard pipeline | implement and verify | verified |
| autoionizing lines | standard pipeline | implement and verify | verified |
| He I isotope families and He II | standard pipeline | implement and verify | verified |
| hydrogen fine structure/Stark/merging | standard pipeline | implement and verify | verified |
| text molecular bands | standard pipeline | implement and verify | verified |
| TiO | standard pipeline | implement and verify | verified |
| H2O source compilation | compiler-only; omitted by standard `_compile_molecular` | implement compiler parity and test runtime omission | verified |
| H3+ | mass metadata exists, but no standard source compiler or pipeline wiring | document as absent from the public standard synthesis path | boundary |
| PRD routing code | catalog route exists; LTE line treatment remains in scope | exact route-ledger audit | verified |
| optional spectral operator | interface only; fitter out of scope | explain boundary and normalization order | boundary |
| `torch.compile` | absent | do not invent | boundary |

## Public-Routine Inventory

Module coverage is necessary but not sufficient. The machine-readable inventory
now exists:

- `audit/paynezero_symbols.json` is the raw AST public surface;
- `scripts/build_symbol_coverage.py` joins that surface to this ledger and
  applies the reviewed semantic-ownership registry;
- `audit/paynezero_symbol_coverage.json` contains 58 modules and 1,501
  public-object records (1,443 unique qualified names);
- all 1,501 records currently have `reviewed_symbol_override` precision; no
  module-default record remains.

Each record carries its primary/supporting location, semantic disposition,
responsibility, gate, source spelling and hashes, alias target where relevant,
and status. `tests/test_symbol_coverage.py` fail-closes module/symbol/default
drift and semantic-ownership mutations.

This is an implemented and reviewed inventory, not a blanket completeness
claim. The reconciled machine ledger contains 401 `integrated`, 926
`verified`, 81 `implemented`, 32 intentionally `planned`, and 61 `boundary`
records. Seven exact test-evidence promotion groups cover 105 defining
symbols; aliases inherit their target's status. Chapter 1 authority is rebound
to the current executed notebook, and a fresh ledger regeneration is
byte-identical.

The remaining 32 `planned` records are explicit work, not unreconciled bulk:
installer behavior, four loader option matrices and aliases, full
convergence/publication surfaces, the synthesis CLI, prewarm routes, atomic
parser/cache option duplicates, and `Spectrum.save_npz`. The “no unmapped
public symbols” gate is verified; final whole-book acceptance remains open
until those owned gaps and the remaining physical/visual gates close.

## End-to-End Gates

| Gate | Required evidence | Status |
| --- | --- | --- |
| data identity | manifests/hashes/roles plus read-only Chapter 6 and Chapter 15 product verification | verified |
| schema v4 | arrays, shapes, units, monotonicity, compatibility | verified |
| fixed-column quantization | round trip and terminal-boundary impact | verified |
| atomic EOS | scalar, atmosphere, synthesis, fixed-ne | integrated |
| molecular state | warm/cool, atmosphere/synthesis catalog distinction | integrated |
| continuum | per-process absorption/scattering and interpolation | integrated |
| line catalogs | compact decode, corrections, routes, selection, counts | verified |
| line opacity | compact ordinary, helium, autoionizing, hydrogen, molecular paths | verified |
| transfer | absorption, scattering, total/continuum, saturated core | verified |
| one atmosphere pass | exact Ch. 11 opacity → Ch. 12 finalization → Ch. 13 remap | verified |
| convergence | stopping state machine plus exact solar trajectory accepted on pass four; three other regimes unrun | verified |
| final atmosphere | exact 80-layer solar schema-v4 atmosphere with three-run byte-identical parity; flux/hydrostatic/optical acceptance remains separate | verified |
| same-atmosphere spectra | compact hot, solar, giant, cool fixtures | verified |
| initialized-label spectra | exact public solar label call plus immutable exploratory safety flags; full four-case label ladder unrun | verified |
| converged workflow | solar labels → initializer → four-pass exact atmosphere → schema-v4 → spectrum; staged/repeat/pinned physical arrays identical | verified |
| cache | cold/warm equality and provenance on covered compact paths | verified |
| backend/thread | CPU/fixed-thread evidence; optional hardware and cross-thread table incomplete | implemented |
| whole-book coverage | all 1,501 symbols mapped; 1,327 integrated/verified, 81 implemented, 32 intentionally planned, 61 boundary; final physical/render gates remain open | implemented |
